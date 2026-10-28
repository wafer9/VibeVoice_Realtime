#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stream paired JSONL into self-contained binary latent or audio TARs.

Globally shuffles complete records with a fixed seed before splitting TARs.
Copies original latent/audio bytes; no base64, resampling, or re-encoding.
The existing three positional arguments and shard/thread options are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import hashlib
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twinlakes.dataset.audio_shard import AUDIO_SHARD_FORMAT, resolve_audio_path
from twinlakes.dataset.latent_shard import (LATENT_SHARD_FORMAT, codec_signature,
                                          load_codec_metadata, load_latent_bytes, resolve_latent_path)


class DigestWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        written = self.stream.write(data)
        self.digest.update(data[:written])
        return written

    def __getattr__(self, name):
        return getattr(self.stream, name)


def add_member(tar, name, source, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    tar.addfile(info, source)


def initialize_pack_worker():
    # Each process validates tiny CPU tensors; avoid one OpenMP pool per process.
    import torch
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)d %(levelname)s %(message)s')


def write_shard(rows, path, roots, index):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    counts, seconds, audio_bytes, seen = {}, 0.0, 0, set()
    with temporary.open("xb", buffering=1024 * 1024) as output:
        hashed = DigestWriter(output)
        with tarfile.open(fileobj=hashed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for row in rows:
                required = {"id", "wav", "text", "prompt_wav", "prompt_text"}
                if not isinstance(row, dict) or not required <= row.keys():
                    raise ValueError("Expected paired JSONL with id/wav/text/prompt_wav/prompt_text")
                key = row["id"]
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                    raise ValueError(f"Invalid TAR sample key: {key!r}")
                if key in seen:
                    raise ValueError(f"Duplicate sample ID within shard: {key}")
                seen.add(key)
                language = row.get("language", key.split("_")[0].lower())
                if language not in roots:
                    raise ValueError(f"Unsupported language: {language}")
                if not isinstance(row["text"], str) or not isinstance(row["prompt_text"], str):
                    raise TypeError(f"Target and prompt text must be strings: {key}")
                target = resolve_audio_path(row["wav"], language, roots)
                prompt = resolve_audio_path(row["prompt_wav"], language, roots)
                if target.resolve() == prompt.resolve():
                    raise ValueError(f"Self-prompt: {key}")
                formats = {"target": target.suffix[1:].lower(), "prompt": prompt.suffix[1:].lower()}
                members = {"target": f"{key}.{formats['target']}", "prompt": f"{key}.prompt"}
                metadata = {**row, "shard_format": AUDIO_SHARD_FORMAT,
                            "audio_members": members, "audio_formats": formats}
                encoded = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
                # Open both files before adding any part of this sample.
                with ExitStack() as stack:
                    files = [stack.enter_context(p.open("rb")) for p in [target, prompt]]
                    sizes = [os.fstat(f.fileno()).st_size for f in files]
                    if not all(sizes):
                        raise ValueError(f"Empty audio file: {key}")
                    add_member(tar, f"{key}.json", io.BytesIO(encoded), len(encoded))
                    for name, stream, size in zip(members.values(), files, sizes):
                        add_member(tar, name, stream, size)
                    audio_bytes += sum(sizes)
                counts[language] = counts.get(language, 0) + 1
                seconds += float(row.get("duration", 0))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    result = {"index": index, "path": str(path.resolve()), "samples": len(rows),
              "languages": counts, "target_hours": seconds / 3600,
              "audio_bytes": audio_bytes, "tar_bytes": path.stat().st_size,
              "sha256": hashed.digest.hexdigest()}
    logging.info("Wrote %s (%d pairs)", path, len(rows))
    return result


def write_latent_shard(rows, path, latent_root, codec_metadata, index):
    started = time.monotonic()
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    codec = codec_signature(codec_metadata)
    counts, seen, seconds, latent_bytes = {}, set(), 0., 0
    with temporary.open('xb', buffering=1024 * 1024) as output:
        hashed = DigestWriter(output)
        with tarfile.open(fileobj=hashed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for row in rows:
                if not {'id', 'wav', 'text', 'prompt_wav', 'prompt_text'} <= row.keys():
                    raise ValueError('Expected paired JSONL; use train.jsonl, not raw en/zh.jsonl')
                key = row['id']
                if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', key) or key in seen:
                    raise ValueError(f'Invalid or duplicate sample ID: {key}')
                seen.add(key)
                language = row.get('language', key.split('_')[0].lower())
                if not all(isinstance(row[name], str) and row[name].strip() for name in ('text', 'prompt_text')):
                    raise ValueError(f'Empty/non-string target or prompt text: {key}')
                target = resolve_latent_path(row['wav'], language, latent_root)
                prompt = resolve_latent_path(row['prompt_wav'], language, latent_root)
                if target.stem != key or target == prompt:
                    raise ValueError(f'Target ID mismatch or self-prompt: {key}')
                # Only two small tensors live in memory. Copy original serialization bytes.
                payloads = [p.read_bytes() for p in (target, prompt)]
                tensors = [load_latent_bytes(payload, codec) for payload in payloads]
                members = {'target': key + '.latent', 'prompt': key + '.prompt'}
                metadata = dict(row, shard_format=LATENT_SHARD_FORMAT, latent_codec=codec,
                                latent_members=members,
                                latent_shapes=dict(target=list(tensors[0].shape), prompt=list(tensors[1].shape)))
                encoded = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                add_member(tar, key + '.json', io.BytesIO(encoded), len(encoded))
                for name, payload in zip(members.values(), payloads):
                    add_member(tar, name, io.BytesIO(payload), len(payload))
                counts[language] = counts.get(language, 0) + 1
                seconds += float(row.get('duration', 0))
                latent_bytes += sum(map(len, payloads))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    result = dict(index=index, path=str(path.resolve()), samples=len(rows), languages=counts,
                  target_hours=seconds / 3600, latent_bytes=latent_bytes, tar_bytes=path.stat().st_size,
                  sha256=hashed.digest.hexdigest(), started_monotonic=started,
                  elapsed_seconds=time.monotonic()-started)
    logging.info('Wrote %s (%d latent pairs)', path, len(rows))
    return result


def shuffle_manifest(source, directory, seed, memory):
    """Globally permute complete JSONL records using disk-backed random-key sort."""
    if shutil.which('sort') is None:
        raise RuntimeError('Global shuffle requires GNU sort')
    directory = Path(directory)
    ordered = directory / 'random_order.jsonl'
    rng = random.Random(seed)
    count = 0
    # Keep records together: target/prompt and both transcripts are never split.
    command = ['sort', '--field-separator=\t', '--key=1,1',
               '--buffer-size=' + memory, '--parallel=4',
               '--temporary-directory=' + str(directory), '--output=' + str(ordered)]
    logging.info('Global shuffle: seed=%d, sort memory=%s, temporary directory=%s', seed, memory, directory)
    started = time.monotonic()
    with (directory / 'sort.stderr').open('w+b') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors,
                                   env=dict(os.environ, LC_ALL='C'))
        try:
            try:
                with process.stdin, Path(source).open('rb') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        key = f'{rng.getrandbits(128):032x}\t'.encode('ascii')
                        process.stdin.write(key + line.rstrip(b'\r\n') + b'\n')
                        count += 1
                        if count % 1000000 == 0:
                            logging.info('Global shuffle: assigned keys to %d records', count)
                logging.info('Global shuffle: all %d keys assigned; waiting for external sort', count)
            except BrokenPipeError:
                pass  # Report the actual sort error below (e.g. a full temporary disk).
            if process.wait():
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 4096))
                raise RuntimeError('Global shuffle failed: ' + errors.read().decode('utf-8', errors='replace'))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    logging.info('Global shuffle ready: %d records, %.1f seconds', count, time.monotonic() - started)
    return ordered, count


def chunks(path, size, max_samples, keyed=False):
    chunk, count = [], 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if max_samples and count >= max_samples:
                break
            if not line.strip():
                continue
            chunk.append(json.loads(line.split('\t', 1)[1] if keyed else line))
            count += 1
            if len(chunk) == size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['latent', 'audio'], default='latent')
    parser.add_argument('--latent-root', type=Path, default=Path('/data/joe/data/latent'))
    parser.add_argument('--codec-metadata', type=Path, help='Default: <latent-root>/_codec.json')
    parser.add_argument("--num_utts_per_shard", "--num-utts-per-shard", type=int, default=10000)
    parser.add_argument("--num_threads", "--num-threads", type=int, default=1,
                        help="Concurrent shard writers; at most 2x this many chunks are queued")
    parser.add_argument('--executor', choices=['thread', 'process'], default='thread',
                        help='Use process for CPU-heavy latent validation/TAR packing')
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True,
                        help='Globally shuffle JSONL records before splitting TARs (default: enabled)')
    parser.add_argument('--seed', type=int, default=20261028, help='Global shuffle seed')
    parser.add_argument('--shuffle-temp-dir', type=Path,
                        help='Local disk for external sorting; default: parent of shards_dir')
    parser.add_argument('--shuffle-memory', default='1G', help='GNU sort memory budget, e.g. 1G or 4G')
    parser.add_argument("--prefix", default="shards")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = entire input; otherwise take N records after optional global shuffle")
    parser.add_argument("--en-root", type=Path, default=Path("/nfs-speech-cfs/ASR/opensource/Emilia/EN"))
    parser.add_argument("--zh-root", type=Path, default=Path("/nfs-speech-cfs/ASR/opensource/Emilia/ZH"))
    parser.add_argument("list_file", type=Path, help="Paired JSONL, e.g. data/Emilia/train.jsonl")
    parser.add_argument("shards_dir", type=Path)
    parser.add_argument("shards_list", type=Path)
    args = parser.parse_args(argv)
    if args.num_utts_per_shard <= 0 or args.num_threads <= 0 or args.max_samples < 0:
        parser.error("Shard size and threads must be positive; max-samples must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("Invalid shard prefix")
    source = args.list_file.resolve()
    directory, listing = args.shards_dir.resolve(), args.shards_list.resolve()
    listing_partial = listing.with_name(listing.name + ".partial")
    if listing.exists() or listing_partial.exists() or (directory.exists() and any(directory.iterdir())):
        raise FileExistsError("Use a new shard directory and list path")
    if not source.is_file():
        raise FileNotFoundError(source)
    directory.mkdir(parents=True, exist_ok=True)
    listing.parent.mkdir(parents=True, exist_ok=True)
    roots = {"en": str(args.en_root.resolve()), "zh": str(args.zh_root.resolve())}
    codec_metadata = None
    if args.format == 'latent':
        codec_metadata = load_codec_metadata(args.codec_metadata or args.latent_root / '_codec.json')
        (directory / '_codec.json').write_text(json.dumps(codec_metadata, ensure_ascii=False, indent=2))
    stat = source.stat()
    report_path = directory / "metadata.json"
    report = {"complete": False, "format": LATENT_SHARD_FORMAT if codec_metadata else AUDIO_SHARD_FORMAT,
              "input": {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
              "audio_roots": roots, "audio_processing": "Original file bytes; no base64/resampling/re-encoding",
              "pairing": "Preserves input target/prompt pair and both transcripts",
              "num_utts_per_shard": args.num_utts_per_shard, "num_threads": args.num_threads,
              "max_samples": args.max_samples, "shards": []}
    report['executor'] = args.executor
    report['shuffle'] = dict(enabled=args.shuffle, seed=args.seed if args.shuffle else None,
                             algorithm='random128_external_sort_v1' if args.shuffle else 'input_order',
                             memory=args.shuffle_memory if args.shuffle else None)
    if codec_metadata:
        report.pop('audio_roots')
        report.pop('audio_processing')
        report.update(latent_root=str(args.latent_root.resolve()), latent_codec=codec_metadata,
                      latent_processing='Original raw encoder mean tensor bytes; no base64/casting/scaling')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    started = time.monotonic()
    pool = ProcessPoolExecutor if args.executor == 'process' else ThreadPoolExecutor
    pool_options = dict(max_workers=args.num_threads)
    if args.executor == 'process':
        pool_options.update(mp_context=multiprocessing.get_context('spawn'), initializer=initialize_pack_worker)
    logging.info('Packing %s: %d %s workers, %d pairs/shard', args.format, args.num_threads,
                 args.executor, args.num_utts_per_shard)
    try:
        with ExitStack() as stack:
            packing_source = source
            if args.shuffle:
                temp_parent = (args.shuffle_temp_dir or directory.parent).resolve()
                temp_parent.mkdir(parents=True, exist_ok=True)
                temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix='emilia_shuffle_', dir=temp_parent))
                packing_source, count = shuffle_manifest(source, temporary, args.seed, args.shuffle_memory)
                report['shuffle'].update(input_samples=count, temporary_parent=str(temp_parent))
            executor = stack.enter_context(pool(**pool_options))
            pending = set()
            for index, chunk in enumerate(chunks(packing_source, args.num_utts_per_shard, args.max_samples,
                                                 keyed=args.shuffle)):
                target = directory / f"{args.prefix}_{index:09d}.tar"
                if codec_metadata:
                    pending.add(executor.submit(write_latent_shard, chunk, target, args.latent_root, codec_metadata, index))
                else:
                    pending.add(executor.submit(write_shard, chunk, target, roots, index))
                if len(pending) >= 2 * args.num_threads:
                    ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                    report["shards"].extend(future.result() for future in ready)
            if pending:
                ready, _ = wait(pending)
                report["shards"].extend(future.result() for future in ready)
        if not report["shards"]:
            raise ValueError("No samples found")
        latest = source.stat()
        if (latest.st_size, latest.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Input manifest changed during packing")
        report["shards"].sort(key=lambda row: row["index"])
        report["total_samples"] = sum(row["samples"] for row in report["shards"])
        if args.shuffle:
            expected = min(count, args.max_samples) if args.max_samples else count
            if report['total_samples'] != expected:
                raise RuntimeError(f'Shuffled sample count mismatch: {report["total_samples"]} != {expected}')
        report["total_target_hours"] = sum(row["target_hours"] for row in report["shards"])
        report["tar_bytes"] = sum(row["tar_bytes"] for row in report["shards"])
        with listing_partial.open("x", encoding="utf-8") as stream:
            for shard in report["shards"]:
                stream.write(shard["path"] + "\n")
        os.replace(listing_partial, listing)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"shards_list": str(listing), "samples": report["total_samples"],
                      "shards": len(report["shards"]), "tar_bytes": report["tar_bytes"]}))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
