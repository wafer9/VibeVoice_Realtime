"""Verify TAR byte fidelity, real DataPipe grouping, and failure propagation."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
import torch

from twinlakes.dataset import processor
from twinlakes.dataset.audio_shard import decode_audio_pair, resolve_audio_path
from twinlakes.dataset.datapipes import WenetTarShardDatasetSource

SPEC = importlib.util.spec_from_file_location('make_shard', Path(__file__).resolve().parents[1] / 'tools/make_shard.py')
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class AudioShardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name in ['target.wav', 'reference.wav']:
            with wave.open(str(self.root / name), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000)
                f.writeframes(np.arange(48000, dtype=np.int16).tobytes())
        self.row = {'id': 'EN_B00000_S00000_W000000', 'language': 'en', 'wav': 'target.wav',
                    'prompt_wav': 'reference.wav', 'text': 'Target sentence', 'prompt_text': 'Reference sentence', 'duration': 2.}

    def tearDown(self):
        self.directory.cleanup()

    def test_tar_bytes_and_current_datapipe(self):
        tar = self.root / 'one.tar'
        report = PACK.write_shard([self.row], tar, {'en': self.root}, 0)
        self.assertEqual(hashlib.sha256(tar.read_bytes()).hexdigest(), report['sha256'])
        listing = self.root / 'data.list'; listing.write_text(str(tar) + '\n')
        samples = list(WenetTarShardDatasetSource(str(listing), partition=False, shuffle=False, prefetch=1))
        self.assertEqual(len(samples), 1)
        decoded = decode_audio_pair(samples[0])
        self.assertEqual(decoded['target_audio_bytes'], (self.root / 'target.wav').read_bytes())
        self.assertEqual(decoded['prompt_audio_bytes'], (self.root / 'reference.wav').read_bytes())
        audio = processor.decode_wav_single(samples[0])
        self.assertEqual(audio['prompt_text'], self.row['prompt_text'])
        self.assertEqual(tuple(audio['wav'].shape), (1, 48000))
        self.assertEqual(self.row['wav'], 'target.wav')

    def test_missing_audio_worker_error_is_not_published(self):
        manifest = self.root / 'input.jsonl'
        manifest.write_text(json.dumps({**self.row, 'prompt_wav': 'missing.wav'}) + '\n')
        listing = self.root / 'data.list'
        with self.assertRaises(FileNotFoundError):
            PACK.main(['--format', 'audio', '--num_threads', '2', '--en-root', str(self.root), str(manifest), str(self.root / 'shards'), str(listing)])
        self.assertFalse(listing.exists())
        self.assertFalse(json.loads((self.root / 'shards/metadata.json').read_text())['complete'])

    def test_emilia_id_resolution(self):
        path = resolve_audio_path('ZH_B00001_S00002_W000003', 'zh', {'zh': '/audio'})
        self.assertEqual(str(path), '/audio/ZH_B00001/ZH_B00001_S00002/mp3/ZH_B00001_S00002_W000003.mp3')

    def test_legacy_latent_ignores_auxiliary_metadata(self):
        def encoded(shape):
            stream = io.BytesIO(); np.save(stream, np.zeros(shape, dtype=np.float32))
            return base64.b64encode(stream.getvalue()).decode()
        obj = {'id': 'x', 'text': 'target', 'prompt_text': 'prompt', 'sim': .9,
               'prompt_emb': 'unused', 'tts_emb': 'unused',
               'tts_latent': encoded((8, 64)), 'prompt_latent': encoded((4, 64))}
        row = processor.decode_latent_single({'key': 'x', 'text': json.dumps(obj)})
        self.assertNotIn('spk_emb', row)
        self.assertNotIn('spk_sim', row)
        self.assertEqual(tuple(row['wav'].shape), (8, 64))


if __name__ == '__main__':
    unittest.main()
