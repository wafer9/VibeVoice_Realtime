"""Binary paired latent byte fidelity, pairing, codec checks, and padding."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from twinlakes.dataset import processor
from twinlakes.dataset.datapipes import WenetTarShardDatasetSource
from twinlakes.dataset.latent_shard import resolve_latent_path

SPEC = importlib.util.spec_from_file_location('pack_latent', Path(__file__).resolve().parents[1] / 'tools/make_shard.py')
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class LatentShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / 'latent'
        self.cache.mkdir()
        self.codec = dict(encoder_sha256='0'*64, implementation_sha256='1'*64,
                          sample_rate=24000, channels=1, dimension=64, hop_length=3200,
                          dtype='torch.bfloat16', latent_mode='encoder mean, no posterior sampling',
                          scaled=False, speech_scaling_factor=.1962890625, speech_bias_factor=-.04931640625,
                          normalization=dict(target_dB_FS=-25, eps=1e-6, avoid_clipping=True))
        (self.cache / '_codec.json').write_text(json.dumps(self.codec))
        self.rows = []
        for language in ('en', 'zh'):
            speaker = language.upper() + '_B00000_S00000'
            reference = speaker + '_W000000'
            for index in range(3):
                key = speaker + f'_W{index:06d}'
                path = resolve_latent_path(key, language, self.cache)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.full((8 + index, 64), index + .25, dtype=torch.bfloat16), path)
                if index:
                    self.rows.append(dict(id=key, language=language,
                                          wav=f'{speaker.split("_S")[0]}/{speaker}/mp3/{key}.mp3',
                                          text=f'target {language} {index}', prompt_wav=reference,
                                          prompt_text=f'prompt {language}', duration=(8+index)/7.5))

    def tearDown(self):
        self.temporary.cleanup()

    def test_binary_tar_roundtrip_preserves_bf16_and_pair(self):
        manifest = self.root / 'train.jsonl'
        manifest.write_text('\n'.join(map(json.dumps, self.rows)) + '\n')
        listing = self.root / 'train.list'
        report = PACK.main(['--latent-root', str(self.cache), '--num-utts-per-shard', '2',
                            '--num-threads', '2', str(manifest), str(self.root/'shards'), str(listing)])
        self.assertEqual(report['total_samples'], 4)
        self.assertTrue(report['complete'])
        self.assertTrue(report['shuffle']['enabled'])
        self.assertEqual(report['shuffle']['input_samples'], 4)
        self.assertFalse(list(self.root.glob('emilia_shuffle_*')))
        samples = list(WenetTarShardDatasetSource(str(listing), partition=False, shuffle=False, prefetch=1))
        self.assertEqual(len(samples), 4)
        decoded = []
        rows_by_id = {row['id']: row for row in self.rows}
        self.assertEqual({sample['key'] for sample in samples}, set(rows_by_id))
        for sample in samples:
            row = rows_by_id[sample['key']]
            original = resolve_latent_path(row['wav'], row['language'], self.cache)
            prompt = resolve_latent_path(row['prompt_wav'], row['language'], self.cache)
            self.assertEqual(sample['latent'], original.read_bytes())
            self.assertEqual(sample['prompt'], prompt.read_bytes())
            data = processor.decode_latent_single(sample, expected_codec=self.codec)
            self.assertTrue(torch.equal(data['wav'], torch.load(original, weights_only=True)))
            self.assertEqual(data['wav'].dtype, torch.bfloat16)
            self.assertEqual(data['prompt_text'], row['prompt_text'])
            self.assertEqual(data['text'], row['text'])
            data.update(input_ids=torch.arange(12), label_ids=torch.arange(12), audio_pos=torch.tensor([1,3,5,11]))
            decoded.append(data)
        padded = processor.padding(decoded)
        self.assertEqual(padded['wavs'].dtype, torch.bfloat16)
        self.assertEqual(tuple(padded['wavs'].shape), (4, 10, 64))
        self.assertEqual(padded['wavs_lengths'].tolist(),
                         [9 if sample['key'].endswith('000001') else 10 for sample in samples])
        bad = copy.deepcopy(self.codec); bad['encoder_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'differs'):
            processor.decode_latent_single(samples[0], expected_codec=bad)

    def test_missing_latent_fails_pack_and_never_publishes_list(self):
        manifest = self.root / 'train.jsonl'
        row = dict(self.rows[0], prompt_wav='EN_B00000_S00000_W999999')
        manifest.write_text(json.dumps(row) + '\n')
        listing = self.root / 'train.list'
        with self.assertRaises(FileNotFoundError):
            PACK.main(['--latent-root', str(self.cache), str(manifest), str(self.root/'shards'), str(listing)])
        self.assertFalse(listing.exists())
        self.assertFalse(json.loads((self.root/'shards/metadata.json').read_text())['complete'])
        self.assertFalse(list(self.root.glob('emilia_shuffle_*')))

    def test_global_shuffle_is_lossless_repeatable_and_mixes_languages(self):
        rows = [dict(id=f'{language}_{index}', language=language,
                     text='中文 \\t target', prompt_text='reference', prompt_wav=f'prompt_{language}_{index}')
                for language in ('en', 'zh') for index in range(512)]
        manifest = self.root / 'grouped.jsonl'
        original = '\n\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n'
        manifest.write_text(original)
        results = []
        for seed in (20261028, 20261028, 42):
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                shuffled, count = PACK.shuffle_manifest(manifest, temporary, seed, '1M')
                blocks = list(PACK.chunks(shuffled, 32, 0, keyed=True))
                result = [row for block in blocks for row in block]
                self.assertEqual(count, len(rows))
                self.assertEqual(len(result), len(rows))
                self.assertEqual({row['id']: row for row in result}, {row['id']: row for row in rows})
                self.assertTrue(all({row['language'] for row in block} == {'en', 'zh'} for block in blocks))
                subset = list(PACK.chunks(shuffled, 7, 13, keyed=True))
                self.assertEqual([row for block in subset for row in block], result[:13])
                results.append(result)
        self.assertEqual(results[0], results[1])
        self.assertNotEqual(results[0], results[2])
        self.assertNotEqual(results[0], rows)
        self.assertEqual(manifest.read_text(), original)

    def test_no_shuffle_preserves_input_order(self):
        manifest = self.root / 'ordered.jsonl'
        manifest.write_text('\n'.join(map(json.dumps, self.rows)) + '\n')
        listing = self.root / 'ordered.list'
        report = PACK.main(['--no-shuffle', '--latent-root', str(self.cache),
                            '--num-utts-per-shard', '2', str(manifest),
                            str(self.root/'ordered_shards'), str(listing)])
        samples = list(WenetTarShardDatasetSource(str(listing), partition=False, strict=True))
        self.assertEqual([sample['key'] for sample in samples], [row['id'] for row in self.rows])
        self.assertFalse(report['shuffle']['enabled'])

    def test_sort_failure_cleans_temporary_files_and_never_publishes_list(self):
        manifest = self.root / 'sort_failure.jsonl'
        manifest.write_text(json.dumps(self.rows[0]) + '\n')
        listing = self.root / 'sort_failure.list'
        with self.assertRaisesRegex(RuntimeError, 'Global shuffle failed'):
            PACK.main(['--shuffle-memory', 'invalid', '--latent-root', str(self.cache),
                       str(manifest), str(self.root/'sort_failure'), str(listing)])
        self.assertFalse(listing.exists())
        self.assertFalse(list(self.root.glob('emilia_shuffle_*')))
        self.assertFalse(json.loads((self.root/'sort_failure/metadata.json').read_text())['complete'])

    def test_invalid_and_scaled_features_rejected(self):
        row = self.rows[0]
        path = resolve_latent_path(row['wav'], 'en', self.cache)
        torch.save(torch.ones(9,64,dtype=torch.float32), path)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            PACK.write_latent_shard([row], self.root/'bad.tar', self.cache, self.codec, 0)
        self.assertFalse((self.root/'bad.tar').exists())
        scaled = dict(self.codec, scaled=True)
        with self.assertRaisesRegex(ValueError, 'raw'):
            PACK.write_latent_shard([row], self.root/'scaled.tar', self.cache, scaled, 0)

    def test_paths_and_frame_filter_boundary(self):
        row = self.rows[0]
        absolute = '/nfs-speech-cfs/ASR/opensource/Emilia/EN/' + row['wav']
        self.assertEqual(resolve_latent_path(absolute, 'en', self.cache),
                         resolve_latent_path(row['wav'], 'en', self.cache))
        with self.assertRaises(ValueError):
            resolve_latent_path(row['prompt_wav'], 'zh', self.cache)
        with self.assertRaises(ValueError):
            resolve_latent_path('../'+row['wav'], 'en', self.cache)
        sample = dict(wav=torch.zeros(225,64), is_latent=True, text='a', prompt_text='b')
        self.assertTrue(processor.filter(sample))
        self.assertFalse(processor.filter(dict(sample, wav=torch.zeros(226,64))))

    def test_missing_tar_is_fatal_in_strict_training_reader(self):
        listing = self.root / 'missing.list'
        listing.write_text(str(self.root/'missing.tar')+'\n')
        with self.assertRaises(processor.UrlOpenError):
            list(WenetTarShardDatasetSource(str(listing), partition=False, strict=True))

    def test_process_pool_cli_can_pack_and_decode(self):
        manifest = self.root / 'process.jsonl'
        manifest.write_text('\n'.join(map(json.dumps, self.rows))+'\n')
        listing = self.root / 'process.list'
        command = [sys.executable, str(Path(PACK.__file__)), '--executor', 'process',
                   '--num_threads', '2', '--num_utts_per_shard', '2', '--latent-root', str(self.cache),
                   str(manifest), str(self.root/'process_shards'), str(listing)]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env=dict(os.environ, OMP_NUM_THREADS='1'))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        samples = list(WenetTarShardDatasetSource(str(listing), partition=False, strict=True))
        self.assertEqual(len(samples), 4)
        self.assertEqual({s['key'] for s in samples}, {r['id'] for r in self.rows})
        for sample in samples:
            self.assertEqual(processor.decode_latent_single(sample)['wav'].dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
