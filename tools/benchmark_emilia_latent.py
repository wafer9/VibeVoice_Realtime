#!/usr/bin/env python3
"""Bounded GPU extraction benchmark; writes only to its isolated output directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--limit', type=int, default=256, help='Positive sample limit per language')
    parser.add_argument('--cases', nargs='+', default=['p1_w4', 'p1_w16', 'p2_w4', 'p4_w4'])
    parser.add_argument('--write-workers', type=int, default=0)
    parser.add_argument('--languages', nargs='+', choices=['en', 'zh'], default=['en', 'zh'])
    parser.add_argument('--partition-mode', choices=['row', 'block'], default='row')
    parser.add_argument('--partition-block-size', type=int, default=1000)
    parser.add_argument('--cache-write-dirs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--barrier-dir', type=Path)
    parser.add_argument('--barrier-hosts', type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1 or not all(re.fullmatch(r'p[1-9][0-9]*_w[1-9][0-9]*', name) for name in args.cases):
        parser.error('Use a positive sample limit and cases like p4_w4')
    gpu_count = len(args.gpu.split(','))
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(host=os.uname().nodename, gpu=args.gpu, limit_per_language=args.limit,
                  python=os.sys.executable, cases=[], complete=False)

    # Warm the same source bytes for every case, so later cases do not get an
    # unfair benefit from the first case's NFS page-cache misses.
    for language in args.languages:
        with (root / f'data/Emilia/{language}.jsonl').open() as source:
            for i, line in enumerate(source):
                if i >= args.limit:
                    break
                row = json.loads(line)
                audio = Path('/nfs-speech-cfs/ASR/opensource/Emilia') / language.upper() / row['wav']
                audio.read_bytes()

    for name in args.cases:
        processes, workers = [int(value) for value in re.fullmatch(r'p([0-9]+)_w([0-9]+)', name).groups()]
        directory = args.output / name
        log_path = args.output / (name + '.log')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                           EXTRACT_PYTHON=os.sys.executable, PROCESSES_PER_GPU=str(processes),
                           OMP_NUM_THREADS='1')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'TORCHELASTIC_RUN_ID'):
            environment.pop(key, None)
        barrier = None
        if args.barrier_dir:
            barrier_path = args.barrier_dir / name
            barrier_path.mkdir(parents=True, exist_ok=True)
        else:
            barrier = tempfile.TemporaryDirectory(prefix='emilia_bench_')
            barrier_path = Path(barrier.name)
        environment['EXTRACT_BENCH_START_BARRIER'] = str(barrier_path)
        environment['EXTRACT_BENCH_BARRIER_HOSTS'] = str(args.barrier_hosts)
        command = ['bash', str(root / 'tools/extract_emilia_latent.sh'),
                   '--limit', str(args.limit), '--output', str(directory),
                   '--write-workers', str(args.write_workers),
                   '--languages', *args.languages,
                   '--partition-mode', args.partition_mode,
                   '--partition-block-size', str(args.partition_block_size),
                   '--cache-write-dirs' if args.cache_write_dirs else '--no-cache-write-dirs',
                   '--workers', str(workers), '--log-interval', '100']
        samples = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                query = subprocess.run(
                    ['nvidia-smi', '-i', args.gpu,
                     '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True)
                if query.returncode == 0:
                    for line in query.stdout.strip().splitlines():
                        gpu, util, memory = [float(x) for x in line.split(',')]
                        samples.append(dict(time=time.time(), gpu=int(gpu), utilization=util, memory_mib=memory))
                stop.wait(.5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        start = time.monotonic()
        print('START', name, flush=True)
        try:
            with log_path.open('x') as log:
                result = subprocess.run(command, env=environment, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        finally:
            stop.set()
            thread.join()
            if barrier:
                barrier.cleanup()
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {log_path}')
        stats = [json.loads(p.read_text()) for p in sorted((directory / '_runs').glob('*/rank-*.json'))]
        assert len(stats) == processes * gpu_count and all(s['complete'] for s in stats)
        assert sum(s['written'] for s in stats) == args.limit * len(args.languages)
        starts, ends = [], []
        for line in log_path.read_text().splitlines():
            match = re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*? (START|END) ', line)
            if match:
                stamp = datetime.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S,%f').timestamp()
                (starts if match[2] == 'START' else ends).append(stamp)
        active_seconds = max(ends) - min(starts)
        active_samples = [s for s in samples if min(starts) <= s['time'] <= max(ends)]
        count = sum(s['written'] for s in stats)
        seconds = sum(s['audio_seconds'] for s in stats)
        total_rank_seconds = sum(s['elapsed_seconds'] for s in stats)
        case = dict(name=name, processes_per_gpu=processes, decode_workers_per_process=workers,
                    write_workers=args.write_workers,
                    count=count, active_seconds=active_seconds, files_per_second=count/active_seconds,
                    audio_seconds_per_second=seconds/active_seconds,
                    wall_seconds=time.monotonic()-start,
                    mean_gpu_utilization=sum(s['utilization'] for s in active_samples)/max(1,len(active_samples)),
                    gpu_utilization_by_device={gpu: sum(s['utilization'] for s in active_samples if s.get('gpu')==int(gpu)) /
                                               max(1,sum(s.get('gpu')==int(gpu) for s in active_samples))
                                               for gpu in args.gpu.split(',')},
                    peak_memory_mib=max(s['memory_mib'] for s in samples),
                    timing_fractions={key: sum(s[key] for s in stats)/total_rank_seconds
                                      for key in ['input_wait_seconds','encode_and_copy_seconds','write_seconds','write_wait_seconds']},
                    gpu_samples=samples, ranks=stats)
        report['cases'].append(case)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps({k:v for k,v in case.items() if k not in ('gpu_samples','ranks')}), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
