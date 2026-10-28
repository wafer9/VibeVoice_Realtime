#!/usr/bin/env python3
"""GPU-only Emilia acoustic encoder mean cache, one torch Tensor per .latent.

Run with the vibe Python, or torchrun for independent workers on each GPU.
Only acoustic encoder weights are loaded; no LLM or codec decoder is needed.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages', nargs='+', choices=['en', 'zh'], default=['en', 'zh'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/Emilia')
    parser.add_argument('--output', type=Path, default=Path('/data/joe/data/latent'),
                        help='Local cache root; transfer to CFS after extraction')
    parser.add_argument('--codec', type=Path,
                        default=Path('/nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B'))
    parser.add_argument('--en-root', type=Path, default=Path('/nfs-speech-cfs/ASR/opensource/Emilia/EN'))
    parser.add_argument('--zh-root', type=Path, default=Path('/nfs-speech-cfs/ASR/opensource/Emilia/ZH'))
    parser.add_argument('--dtype', choices=['bf16', 'fp32', 'fp16'], default='bf16',
                        help='Physical encoder/input and saved tensor dtype')
    parser.add_argument('--workers', type=int, default=4, help='Bounded audio decode threads per process')
    parser.add_argument('--write-workers', type=int, default=0,
                        help='Async latent write threads per process; 0=synchronous')
    parser.add_argument('--partition-mode', choices=['row', 'block'], default='row')
    parser.add_argument('--partition-block-size', type=int, default=1000)
    parser.add_argument('--cache-write-dirs', action=argparse.BooleanOptionalAction, default=True,
                        help='Cache prepared output parents to avoid repeated directory operations')
    parser.add_argument('--threads', type=int, default=1, help='Torch CPU threads per encoder process')
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only first N nonblank records PER language, 0=all')
    parser.add_argument('--verify-existing', action='store_true',
                        help='Load and validate every existing tensor before skipping (slower)')
    args = parser.parse_args(argv)
    if min(args.workers, args.threads, args.log_interval, args.partition_block_size) < 1 or min(args.limit, args.write_workers) < 0:
        parser.error('workers/threads/log-interval must be positive; limit/write-workers must be nonnegative')
    args.languages = list(dict.fromkeys(args.languages))
    return args


def worker_device_index(local_rank, visible_devices, processes_per_gpu):
    if processes_per_gpu < 1 or not 0 <= local_rank < visible_devices * processes_per_gpu:
        raise ValueError('Invalid local worker rank, visible GPU count, or PROCESSES_PER_GPU')
    return local_rank // processes_per_gpu


def require_cuda(local_rank, processes_per_gpu=1):
    # Deliberately no CPU fallback: extraction was explicitly requested on GPU.
    if not torch.cuda.is_available():
        raise RuntimeError('GPU extraction requires accessible /dev/nvidia* and CUDA. '
                           'torch.cuda.is_available() is False; CPU fallback is disabled.')
    index = worker_device_index(local_rank, torch.cuda.device_count(), processes_per_gpu)
    torch.cuda.set_device(index)
    return torch.device('cuda', index)


def audio_paths(row, language, roots, output):
    """Keep the manifest's relative directory tree, including the mp3 directory."""
    if row.get('language', language).lower() != language:
        raise ValueError('Manifest language mismatch')
    source = Path(row['wav'])
    relative = source.relative_to(roots[language]) if source.is_absolute() else source
    if '..' in relative.parts or not relative.parts or relative.suffix.lower() != '.mp3':
        raise ValueError(f'Unsafe or non-mp3 Emilia path: {source}')
    if not relative.parts[0].startswith(language.upper() + '_B'):
        raise ValueError(f'Wrong language directory: {relative}')
    if row['id'] != relative.stem:
        raise ValueError(f'Audio ID differs from path: {row["id"]}')
    return roots[language] / relative, output / language.upper() / relative.with_suffix('.latent')


def valid_tensor(tensor, dimension, dtype):
    return (isinstance(tensor, torch.Tensor) and tensor.ndim == 2
            and tensor.shape[0] > 0 and tensor.shape[1] == dimension
            and tensor.dtype == dtype and bool(torch.isfinite(tensor).all()))


def existing_latent(path, verify, dimension, dtype):
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not verify:
        return True
    try:
        return valid_tensor(torch.load(path, map_location='cpu', weights_only=True), dimension, dtype)
    except Exception:
        return False  # Re-extract a broken cache when verification was requested.


def records(args, roots, rank, world_size, dimension, dtype, stats):
    for language in args.languages:
        with (args.data_dir / f'{language}.jsonl').open(encoding='utf-8') as stream:
            count = 0
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if args.limit and count >= args.limit:
                    break
                unit = count // args.partition_block_size if args.partition_mode == 'block' else count
                assigned = unit % world_size == rank
                count += 1
                if not assigned:
                    continue
                # Invalid metadata is fatal, with the source location preserved.
                try:
                    row = json.loads(line)
                    source, destination = audio_paths(row, language, roots, args.output)
                    stats['selected'] += 1
                    if existing_latent(destination, args.verify_existing, dimension, dtype):
                        stats['skipped'] += 1
                        if stats['skipped'] % args.log_interval == 0:
                            logging.info('RESUME selected=%d skipped=%d', stats['selected'], stats['skipped'])
                        continue
                    yield {'id': row['id'], 'source': source, 'destination': destination,
                           'language': language, 'line': line_number}
                except Exception as exc:
                    raise RuntimeError(f'{language}.jsonl:{line_number}: {exc}') from exc


def load_audio(record):
    """Match processor.decode_wav_raw + tokenize: mean channels, resample, normalize."""
    import torchaudio
    from vibevoice.processor.vibevoice_tokenizer_processor import AudioNormalizer
    try:
        waveform, rate = torchaudio.load(str(record['source']))
        waveform = waveform.mean(dim=0, keepdim=True)
        if rate != 24000:
            waveform = torchaudio.functional.resample(waveform, rate, 24000)
        if not waveform.numel() or not bool(torch.isfinite(waveform).all()):
            raise ValueError('Empty or nonfinite audio')
        normalized = AudioNormalizer()(waveform[0].numpy())
        return record, torch.from_numpy(normalized).contiguous(), None
    except Exception as exc:
        return record, None, f'{type(exc).__name__}: {exc}'


def prefetched_audio(source, workers):
    # Executor.map on Python 3.12 eagerly consumes iterables: use an explicit bound.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        source = iter(source)
        for _ in range(workers * 2):
            record = next(source, None)
            if record is None:
                break
            pending.append(pool.submit(load_audio, record))
        while pending:
            yield pending.popleft().result()
            record = next(source, None)
            if record is not None:
                pending.append(pool.submit(load_audio, record))


def load_encoder(codec, dtype, device):
    from safetensors import safe_open
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import TokenizerEncoder

    full_config = json.loads((codec / 'config.json').read_text())
    config = VibeVoiceAcousticTokenizerConfig(**full_config['acoustic_tokenizer_config'])
    if not config.causal or config.pad_mode != 'constant':
        raise ValueError('Expected the current causal, constant-padded TwinLakes codec')
    # Same encoder configuration as VibeVoiceAcousticTokenizerModel.__init__.
    encoder_config = copy.deepcopy(config)
    encoder_config.dimension = config.vae_dim
    encoder_config.n_filters = config.encoder_n_filters
    encoder_config.ratios = config.encoder_ratios
    encoder_config.depths = ([int(x) for x in config.encoder_depths.split('-')]
                             if isinstance(config.encoder_depths, str) else config.encoder_depths)
    encoder_config.norm = config.conv_norm
    encoder_config.bias = config.conv_bias
    encoder = TokenizerEncoder(encoder_config)
    index_file = codec / 'model.safetensors.index.json'
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())['weight_map']
        shards = sorted({name for key, name in weight_map.items()
                         if key.startswith('model.acoustic_tokenizer.encoder.')})
    else:
        shards = [p.name for p in sorted(codec.glob('*.safetensors'))]
    prefix = 'model.acoustic_tokenizer.encoder.'
    state = {}
    factors = {}
    for name in shards:
        with safe_open(str(codec / name), framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = stream.get_tensor(key)
    # Scaling is recorded only: cached features are raw, unscaled encoder means.
    for factor in ['speech_scaling_factor', 'speech_bias_factor']:
        key = 'model.' + factor
        factor_shards = [weight_map[key]] if index_file.exists() else shards
        for name in factor_shards:
            with safe_open(str(codec / name), framework='pt', device='cpu') as stream:
                if key in stream.keys():
                    factors[factor] = stream.get_tensor(key).item()
                    break
        if factor not in factors:
            raise ValueError(f'Missing codec scaling metadata: {key}')
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.requires_grad_(False).eval().to(device=device, dtype=dtype)
    implementation = Path(sys.modules[TokenizerEncoder.__module__].__file__)
    metadata = {
        'format': 'emilia_acoustic_mean_tensor_v1', 'codec': str(codec.resolve()),
        'encoder_sha256': digest.hexdigest(),
        'implementation_sha256': hashlib.sha256(implementation.read_bytes()).hexdigest(),
        'acoustic_tokenizer_config': full_config['acoustic_tokenizer_config'],
        'sample_rate': 24000, 'channels': 1, 'resample': 'torchaudio.functional.resample defaults',
        'normalization': {'target_dB_FS': -25, 'eps': 1e-6, 'avoid_clipping': True},
        'latent_mode': 'encoder mean, no posterior sampling', 'scaled': False,
        'model_space_formula': '(raw_latent + speech_bias_factor) * speech_scaling_factor',
        **factors, 'dtype': str(dtype), 'dimension': config.vae_dim,
        'hop_length': math.prod(config.encoder_ratios),
        'storage': 'torch.save CPU contiguous Tensor [frames, dimension], no batch/padding',
        'path': '<output>/<EN or ZH>/<manifest wav relative path with .mp3 replaced by .latent>',
        'batching': 'one complete utterance per encode; no cross-utterance padding or truncation',
    }
    return encoder, metadata


@lru_cache(maxsize=2048)
def prepare_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)


def atomic_save(destination, save, cache_parent=False):
    if cache_parent:
        prepare_directory(destination.parent)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.partial',
                                        dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            save(stream)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(destination, value):
    atomic_save(destination, lambda stream: stream.write(
        json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')))


class LatentWriter:
    """Bounded CPU-only writer; count files only after successful atomic publication."""
    def __init__(self, stats, workers, cache_directories=True):
        self.stats = stats
        self.cache_directories = cache_directories
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers else None
        self.pending = deque()
        self.limit = max(1, workers * 8)

    @staticmethod
    def write(destination, latent, audio_seconds, cache_directories=True):
        start = time.monotonic()
        atomic_save(destination, lambda stream: torch.save(latent, stream), cache_parent=cache_directories)
        return latent.shape[0], audio_seconds, time.monotonic() - start

    def commit(self, result):
        frames, seconds, elapsed = result
        self.stats['written'] += 1
        self.stats['frames'] += frames
        self.stats['audio_seconds'] += seconds
        self.stats['write_seconds'] += elapsed

    def finish_one(self):
        start = time.monotonic()
        self.commit(self.pending.popleft().result())
        self.stats['write_wait_seconds'] += time.monotonic() - start

    def collect_ready(self):
        while self.pending and self.pending[0].done():
            self.finish_one()

    def submit(self, destination, latent, audio_seconds):
        if self.pool is None:
            start = time.monotonic()
            self.commit(self.write(destination, latent, audio_seconds, self.cache_directories))
            self.stats['write_wait_seconds'] += time.monotonic() - start
        else:
            self.collect_ready()
            if len(self.pending) >= self.limit:
                self.finish_one()
            self.pending.append(self.pool.submit(self.write, destination, latent, audio_seconds, self.cache_directories))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            while self.pending:
                self.finish_one()
        finally:
            if self.pool:
                self.pool.shutdown(wait=True)


def ensure_metadata(output, metadata):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '_codec.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = output / '_codec.json'
        if destination.exists():
            if json.loads(destination.read_text()) != metadata:
                raise ValueError('Existing cache codec/dtype/protocol differs; use a separate output')
        else:
            if next(output.glob('**/*.latent'), None) is not None:
                raise ValueError('Existing .latent files lack codec metadata; use a separate output')
            write_json(destination, metadata)


def main(argv=None):
    args = parse_args(argv)
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not 0 <= rank < world_size:
        raise ValueError('Expected 0 <= RANK < WORLD_SIZE')
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s rank={rank} %(message)s')
    processes_per_gpu = int(os.environ.get('PROCESSES_PER_GPU', '1'))
    device = require_cuda(local_rank, processes_per_gpu)
    torch.set_num_threads(args.threads)
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU does not support BF16; explicitly choose another dtype')
    roots = {'en': args.en_root.absolute(), 'zh': args.zh_root.absolute()}
    for language in args.languages:
        if not (args.data_dir / f'{language}.jsonl').is_file() or not roots[language].is_dir():
            raise FileNotFoundError(f'Missing {language} manifest or audio root')
    encoder, metadata = load_encoder(args.codec, dtype, device)
    metadata['audio_roots'] = {k: str(v) for k, v in roots.items()}
    ensure_metadata(args.output, metadata)
    run_id = os.environ.get('TORCHELASTIC_RUN_ID', str(uuid.uuid4()))
    run_dir = args.output / '_runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / f'rank-{rank:03d}.json'
    stats = dict(rank=rank, world_size=world_size, device=str(device),
                 hostname=os.uname().nodename,
                 local_rank=local_rank, processes_per_gpu=processes_per_gpu,
                 gpu_name=torch.cuda.get_device_name(device), complete=False,
                 selected=0, written=0, skipped=0, failed=0, audio_seconds=0., frames=0,
                 input_wait_seconds=0., encode_and_copy_seconds=0., write_seconds=0.,
                 write_wait_seconds=0., write_workers=args.write_workers, submitted=0,
                 partition_mode=args.partition_mode, partition_block_size=args.partition_block_size,
                 cache_write_dirs=args.cache_write_dirs,
                 languages=args.languages, limit_per_language=args.limit,
                 manifests={lang: str((args.data_dir / f'{lang}.jsonl').absolute()) for lang in args.languages})
    # Used only by the bounded benchmark: exclude uneven model load times.
    barrier = os.environ.get('EXTRACT_BENCH_START_BARRIER')
    if barrier:
        barrier = Path(barrier)
        (barrier / f'{os.uname().nodename}_{rank}.ready').touch()
        expected = world_size * int(os.environ.get('EXTRACT_BENCH_BARRIER_HOSTS', '1'))
        deadline = time.monotonic() + 120
        while len(list(barrier.glob('*.ready'))) < expected:
            if time.monotonic() > deadline:
                raise TimeoutError('Benchmark model-loading barrier timed out')
            time.sleep(.05)
    start = time.monotonic()
    logging.info('START %s', json.dumps(stats))
    write_json(report_path, stats)
    try:
        with (run_dir / f'rank-{rank:03d}.errors.jsonl').open('a', buffering=1) as errors, \
                LatentWriter(stats, args.write_workers, args.cache_write_dirs) as writer:
            source = records(args, roots, rank, world_size, metadata['dimension'], dtype, stats)
            with torch.inference_mode():
                previous_end = time.monotonic()
                for record, audio, error in prefetched_audio(source, args.workers):
                    encode_start = time.monotonic()
                    stats['input_wait_seconds'] += encode_start - previous_end
                    if error is None:
                        # Exactly codec.encode(audio[None, None]).mean: no sample/no scaling.
                        latent = encoder(audio[None, None].to(device=device, dtype=dtype))
                        latent = latent.transpose(1, 2)[0].contiguous().cpu()
                        expected = math.ceil(audio.numel() / metadata['hop_length'])
                        if latent.shape[0] != expected or not valid_tensor(latent, metadata['dimension'], dtype):
                            raise RuntimeError(f'Invalid encoder output for {record["id"]}: {latent.shape}')
                        write_start = time.monotonic()
                        stats['encode_and_copy_seconds'] += write_start - encode_start
                        # Thread errors propagate on collection/drain, before marking complete.
                        writer.submit(record['destination'], latent, audio.numel() / 24000)
                        stats['submitted'] += 1
                    else:
                        stats['failed'] += 1
                        errors.write(json.dumps({**{k: str(v) for k, v in record.items()},
                                                 'error': error}, ensure_ascii=False) + '\n')
                        logging.error('%s %s', record['id'], error)
                    if (stats['submitted'] + stats['failed']) % args.log_interval == 0:
                        writer.collect_ready()
                        stats['elapsed_seconds'] = time.monotonic() - start
                        write_json(report_path, stats)
                        logging.info('PROGRESS %s', json.dumps(stats))
                    previous_end = time.monotonic()
        stats['complete'] = stats['failed'] == 0
    finally:
        stats['elapsed_seconds'] = time.monotonic() - start
        write_json(report_path, stats)
        logging.info('END %s', json.dumps(stats))
    if stats['failed']:
        raise RuntimeError(f'{stats["failed"]} audio files failed; see {run_dir}. Rerun to retry.')


if __name__ == '__main__':
    main()
