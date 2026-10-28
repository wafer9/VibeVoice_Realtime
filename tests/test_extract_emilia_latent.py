"""Check the million-file extraction protocol without requiring CUDA for I/O checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location(
    'extract_emilia_latent', Path(__file__).resolve().parents[1] / 'tools/extract_emilia_latent.py')
EXTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACT)


class ExtractionTests(unittest.TestCase):
    def test_selected_language_does_not_open_other_manifest(self):
        for language in ('en', 'zh'):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                key = language.upper() + '_B00000_S00000_W000000'
                row = dict(id=key, language=language,
                           wav=f'{language.upper()}_B00000/{key.rsplit("_W", 1)[0]}/mp3/{key}.mp3')
                (directory / f'{language}.jsonl').write_text(json.dumps(row) + '\n')
                args = EXTRACT.parse_args(['--languages', language, '--data-dir', str(directory),
                                           '--output', str(directory / 'latent')])
                stats = dict(selected=0, skipped=0)
                records = list(EXTRACT.records(args, {'en': Path('/audio/EN'), 'zh': Path('/audio/ZH')},
                                               0, 1, 64, torch.bfloat16, stats))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['language'], language)
                self.assertEqual(records[0]['destination'].relative_to(args.output).parts[0], language.upper())

    def test_launcher_positional_language_forwards_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fake = directory / 'python'
            argv_path = directory / 'argv.json'
            fake.write_text(f'#!{sys.executable}\nimport json, os, pathlib, sys\n'
                            'if sys.argv[1] == "-c": print(1)\n'
                            'else: pathlib.Path(os.environ["ARGV_PATH"]).write_text(json.dumps(sys.argv[2:]))\n')
            fake.chmod(0o755)
            launcher = Path(__file__).resolve().parents[1] / 'tools/extract_emilia_latent.sh'
            for language in ('en', 'zh'):
                subprocess.run(['bash', str(launcher), language, '--workers', '8'], check=True,
                               env=dict(os.environ, EXTRACT_PYTHON=str(fake), PROCESSES_PER_GPU='1', ARGV_PATH=str(argv_path)))
                self.assertEqual(json.loads(argv_path.read_text()), ['--workers', '8', '--languages', language])

    def test_paths_preserve_directory_tree_and_reject_escape(self):
        roots = {'en': Path('/audio/EN'), 'zh': Path('/audio/ZH')}
        for language in ('en', 'zh'):
            key = language.upper() + '_B00000_S00001_W000003'
            relative = Path(key.split('_S')[0]) / key.split('_W')[0] / 'mp3' / (key + '.mp3')
            row = {'id': key, 'wav': str(relative), 'language': language}
            for wav in (str(relative), str(roots[language] / relative)):
                source, output = EXTRACT.audio_paths({**row, 'wav': wav}, language, roots, Path('/latent'))
                self.assertEqual(source, roots[language] / relative)
                self.assertEqual(output, Path('/latent') / language.upper() / relative.with_suffix('.latent'))
            with self.assertRaises(ValueError):
                EXTRACT.audio_paths({**row, 'wav': '../' + str(relative)}, language, roots, Path('/latent'))
            with self.assertRaises(ValueError):
                EXTRACT.audio_paths({**row, 'wav': '/outside/' + str(relative)}, language, roots, Path('/latent'))

    def test_rank_partition_limit_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for language in ('en', 'zh'):
                rows = []
                for i in range(9):
                    key = f'{language.upper()}_B00000_S00000_W{i:06d}'
                    rows.append(dict(id=key, wav=f'{language.upper()}_B00000/{key.rsplit("_W", 1)[0]}/mp3/{key}.mp3'))
                (directory / f'{language}.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
            args = EXTRACT.parse_args(['--data-dir', str(directory), '--output', str(directory / 'latent'),
                                       '--limit', '7', '--verify-existing'])
            partitions = []
            roots = {'en': Path('/audio/EN'), 'zh': Path('/audio/ZH')}
            for rank in range(3):
                stats = dict(selected=0, skipped=0)
                partitions.append(list(EXTRACT.records(args, roots, rank, 3, 64, torch.bfloat16, stats)))
            keys = [record['id'] for partition in partitions for record in partition]
            self.assertEqual(len(keys), 14)
            self.assertEqual(len(set(keys)), 14)
            record = partitions[0][0]
            EXTRACT.atomic_save(record['destination'], lambda f: torch.save(torch.zeros(4, 64, dtype=torch.bfloat16), f))
            stats = dict(selected=0, skipped=0)
            remaining = list(EXTRACT.records(args, roots, 0, 3, 64, torch.bfloat16, stats))
            self.assertEqual(stats['skipped'], 1)
            self.assertEqual(len(remaining), len(partitions[0]) - 1)
            record['destination'].write_bytes(b'broken tensor')
            stats = dict(selected=0, skipped=0)
            self.assertEqual(len(list(EXTRACT.records(args, roots, 0, 3, 64, torch.bfloat16, stats))), len(partitions[0]))

    def test_block_partition_covers_tail_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = []
            for i in range(11):
                key = f'EN_B00000_S{i:05d}_W000000'
                rows.append(dict(id=key, wav=f'EN_B00000/{key.rsplit("_W", 1)[0]}/mp3/{key}.mp3'))
            (directory / 'en.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
            args = EXTRACT.parse_args(['--data-dir', str(directory), '--languages', 'en',
                                      '--output', str(directory / 'latent'),
                                      '--partition-mode', 'block', '--partition-block-size', '3'])
            partitions = []
            for rank in range(5):
                stats = dict(selected=0, skipped=0)
                partitions.append(list(EXTRACT.records(args, {'en': Path('/audio/EN')}, rank, 5, 64, torch.bfloat16, stats)))
            ids = [record['id'] for partition in partitions for record in partition]
            self.assertEqual(set(ids), {row['id'] for row in rows})
            self.assertEqual(len(ids), 11)
            self.assertEqual([len(p) for p in partitions], [3, 3, 3, 2, 0])

    def test_atomic_failure_and_codec_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            EXTRACT.ensure_metadata(output, {'scaled': False, 'encoder_sha256': 'first'})
            EXTRACT.ensure_metadata(output, {'scaled': False, 'encoder_sha256': 'first'})
            with self.assertRaises(ValueError):
                EXTRACT.ensure_metadata(output, {'scaled': False, 'encoder_sha256': 'second'})
            destination = output / 'target.latent'
            def fail(stream):
                stream.write(b'incomplete')
                raise OSError('Simulated failed write')
            with self.assertRaises(OSError):
                EXTRACT.atomic_save(destination, fail)
            self.assertFalse(destination.exists())
            self.assertFalse(list(output.glob('*.partial')))

    def test_cuda_required_and_no_cpu_fallback(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                EXTRACT.require_cuda(0)

    def test_multiple_processes_map_to_each_gpu(self):
        mapping = [EXTRACT.worker_device_index(rank, 8, 2) for rank in range(16)]
        self.assertEqual(mapping, [gpu for gpu in range(8) for _ in range(2)])
        self.assertEqual([EXTRACT.worker_device_index(rank, 8, 1) for rank in range(8)], list(range(8)))
        for values in [(16, 8, 2), (-1, 8, 2), (0, 8, 0)]:
            with self.assertRaises(ValueError):
                EXTRACT.worker_device_index(*values)

    def test_async_writes_drained_and_failures_propagate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stats = dict(written=0, frames=0, audio_seconds=0., write_seconds=0., write_wait_seconds=0.)
            with EXTRACT.LatentWriter(stats, 2) as writer:
                for i in range(20):
                    writer.submit(directory / f'{i}.latent', torch.full((3, 64), i, dtype=torch.bfloat16), .4)
            self.assertEqual(stats['written'], 20)
            self.assertEqual(stats['frames'], 60)
            for i in range(20):
                self.assertTrue(torch.all(torch.load(directory / f'{i}.latent', weights_only=True) == i))
            with patch.object(EXTRACT, 'atomic_save', side_effect=OSError('write failed')):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    with EXTRACT.LatentWriter(stats, 1) as writer:
                        writer.submit(directory / 'failed.latent', torch.zeros(3,64), .4)
            self.assertEqual(stats['written'], 20)
            self.assertFalse((directory / 'failed.latent').exists())


if __name__ == '__main__':
    unittest.main()
