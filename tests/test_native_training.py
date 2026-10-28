"""Exercise the current native trainer without loading pretrained model weights."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
import yaml

from twinlakes.bin.train import get_args, make_loader, model_parameter_summary, log_model_info
from twinlakes.bin.train import main as train_main
from twinlakes.utils.checkpoint import load_checkpoint, save_checkpoint
from twinlakes.utils.executor import Executor, synchronized_batches
from twinlakes.utils.scheduler import WarmupLR


class TinyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.autocast = False
        self.seen_dtypes = {}

    def forward(self, wavs, labels, **kwargs):
        self.autocast = torch.is_autocast_enabled('cpu')
        self.seen_dtypes = {name: value.dtype for name, value in {'wavs': wavs, 'labels': labels, **kwargs}.items() if isinstance(value, torch.Tensor)}
        self.seen_positions = [x.dtype for x in kwargs.get('audio_pos', [])]
        labels.fill_(0)  # The real model also rewrites labels; protect caller data.
        loss = (self.weight * wavs.mean() - 2).square()
        return {'loss': loss, 'loss_text': loss * .1, 'diffusion_loss': loss * .9}


def batch(value=1., size=1):
    return {'keys': ['sample'] * size, 'wavs': torch.full((size, 4, 1), value),
            'prompt_wavs': torch.ones(size, 2, 1), 'wavs_lengths': torch.full((size,), 4),
            'prompt_wavs_lengths': torch.full((size,), 2), 'input_ids': torch.zeros(size, 3, dtype=torch.long),
            'label_ids': torch.full((size, 3), -100, dtype=torch.long), 'audio_pos': []}


class NativeTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_model_summary_counts_tied_weights_and_frozen_codec(self):
        model = torch.nn.Module()
        model.register_parameter('root_weight', torch.nn.Parameter(torch.ones(3)))
        model.lm = torch.nn.ModuleDict(dict(embedding=torch.nn.Embedding(8, 4), output=torch.nn.Linear(4,8,bias=False)))
        model.lm['output'].weight = model.lm['embedding'].weight
        model.acoustic_tokenizer = torch.nn.ModuleDict(dict(encoder=torch.nn.Linear(1,2), decoder=torch.nn.Linear(2,1)))
        # Match the named encoder/decoder properties used by the real codec.
        model.acoustic_tokenizer.requires_grad_(False)
        model.diffusion_head = torch.nn.Linear(4, 4)
        model.to(torch.bfloat16)
        rows = model_parameter_summary(model)
        self.assertEqual(rows['lm']['total'], 32)
        self.assertEqual(rows['acoustic_tokenizer']['frozen'], 7)
        self.assertEqual(rows['acoustic_tokenizer.encoder']['total'], 4)
        self.assertEqual(rows['acoustic_tokenizer.decoder']['total'], 3)
        self.assertEqual(rows['TOTAL'], dict(total=62, trainable=55, frozen=7, dtype='bfloat16'))
        configs = dict(temporal_model='Qwen3-0.6B-Base', vibevoice_path='codec')
        with patch('twinlakes.bin.train.logging.info') as log:
            log_model_info(model, configs, 'active.yaml', rank=1)
            log.assert_not_called()
        with self.assertLogs(level='INFO') as logs:
            log_model_info(model, configs, 'active.yaml', rank=0)
        text = '\n'.join(logs.output)
        self.assertIn('Qwen3-0.6B-Base', text)
        self.assertIn('Actual model structure', text)
        self.assertIn('Parameter summary', text)
        self.assertIn('bfloat16', text)

    def test_accumulation_including_short_final_window(self):
        model = TinyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        sample = batch()
        executor = Executor()
        executor.train(model, optimizer, scheduler, [sample] * 3, None,
                       {'accum_grad': 2, 'grad_clip': 100., 'log_interval': 100})
        self.assertEqual(executor.step, 2)
        self.assertAlmostEqual(model.weight.item(), 1.36, places=6)
        self.assertTrue((sample['label_ids'] == -100).all())

    def test_validation_is_weighted_without_dummy_sample(self):
        model = TinyLoss()
        loss = Executor().cv(model, [batch(1., 1), batch(2., 3)], None, {})
        self.assertAlmostEqual(loss, .25)
        with self.assertRaisesRegex(ValueError, 'no samples'):
            Executor().cv(model, [], None, {})

    def test_cpu_bf16_uses_autocast(self):
        model = TinyLoss().to(torch.bfloat16)
        self.assertAlmostEqual(Executor().cv(model, [batch()], None, {'dtype': 'bf16'}), 1.)
        self.assertTrue(model.autocast)

    def test_executor_casts_float_inputs_and_preserves_integer_inputs(self):
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                model = TinyLoss().to(dtype=dtype)
                sample = batch()
                sample['wavs'] = sample['wavs'].double()
                sample['prompt_wavs'] = sample['prompt_wavs'].half()
                sample['audio_pos'] = [torch.tensor([1, 2], dtype=torch.int32), torch.tensor([True])]
                Executor()._forward(model, sample, dtype)
                for key in ['wavs', 'prompt_wavs']:
                    self.assertEqual(model.seen_dtypes[key], dtype)
                for key in ['input_ids', 'labels', 'wavs_lengths', 'prompt_wavs_lengths']:
                    self.assertEqual(model.seen_dtypes[key], torch.int64)
                self.assertEqual(model.seen_positions, [torch.int32, torch.bool])
                self.assertTrue((sample['label_ids'] == -100).all())

    def test_checkpoint_does_not_restore_old_parameter_dtype(self):
        model = TinyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        (model.weight - 2).square().backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fp32.pt'
            save_checkpoint(model, path, optimizer=optimizer)
            restored = TinyLoss().to(torch.bfloat16)
            opt = torch.optim.Adam(restored.parameters(), lr=.01)
            load_checkpoint(restored, path, optimizer=opt, strict=True)
            self.assertEqual(restored.weight.dtype, torch.bfloat16)
            self.assertEqual(opt.state[restored.weight]['exp_avg'].dtype, torch.bfloat16)

    def test_vibevoice_head_and_uncondition_mlp_match_lm_width(self):
        from twinlakes.models.rq_transformer import LMModel
        from vibevoice.modular.configuration_vibevoice import VibeVoiceDiffusionHeadConfig
        # The source head width differs from the LM, just like 1536 vs 2048.
        head_config = VibeVoiceDiffusionHeadConfig(hidden_size=12, latent_size=4, head_layers=1)
        lm = torch.nn.Module()
        lm.config = SimpleNamespace(hidden_size=16)
        codec = torch.nn.Module()
        codec.config = SimpleNamespace(vae_dim=4)
        original = SimpleNamespace(model=SimpleNamespace(acoustic_tokenizer=codec),
                                   config=SimpleNamespace(diffusion_head_config=head_config),
                                   speech_scaling_factor=torch.tensor(1.), speech_bias_factor=torch.tensor(0.))
        with patch('twinlakes.models.rq_transformer.AutoConfig.from_pretrained', return_value=lm.config), \
             patch('twinlakes.models.rq_transformer.AutoModelForCausalLM.from_pretrained', return_value=lm), \
             patch('twinlakes.models.rq_transformer.VibeVoiceForConditionalGenerationInference.from_pretrained', return_value=original), \
             patch('vibevoice.modular.modular_vibevoice_text_tokenizer.VibeVoiceTextTokenizerFast.from_pretrained', return_value=object()):
            for name, expected_dtype in [('fp32', torch.float32), ('bf16', torch.bfloat16), ('fp16', torch.float16)]:
                model = LMModel.from_audio_text_pretrained({'temporal_model': 'stub', 'vibevoice_path': 'stub', 'diffusion_head_type': 'vibevoice', 'dtype': name})
                self.assertTrue(all(p.dtype == expected_dtype for p in model.parameters()))
                self.assertTrue(all(b.dtype == expected_dtype for b in model.buffers() if b.is_floating_point()))
            model.float()
        self.assertEqual(model.diffusion_head.config.hidden_size, 16)
        self.assertEqual(tuple(model.audio_bos.shape), (1, 1, 4))
        condition = model.uncondition_mlp(torch.randn(2, 4))
        self.assertEqual(tuple(condition.shape), (2, 16))
        prediction = model.diffusion_head(noisy_images=torch.randn(2, 4), timesteps=torch.tensor([1., 2.]), condition=condition)
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_checkpoint_restores_optimizer_and_scheduler(self):
        def create():
            model = TinyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            scheduler = WarmupLR(optimizer, warmup_steps=4)
            return model, optimizer, scheduler
        def update(model, optimizer, scheduler):
            optimizer.zero_grad(); (model.weight - 2).square().backward(); optimizer.step(); scheduler.step()
        model, optimizer, scheduler = create()
        update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '0_1.pt'
            save_checkpoint(model, path, {'step': 1, 'epoch': 0}, optimizer=optimizer, scheduler=scheduler)
            restored, opt2, sched2 = create()
            infos = load_checkpoint(restored, path, optimizer=opt2, scheduler=sched2, strict=True)
            self.assertEqual(infos['step'], 1)
            update(model, optimizer, scheduler); update(restored, opt2, sched2)
            self.assertTrue(torch.equal(model.weight, restored.weight))
            self.assertEqual(scheduler.state_dict(), sched2.state_dict())
            self.assertTrue(torch.equal(optimizer.state[model.weight]['exp_avg'], opt2.state[restored.weight]['exp_avg']))
            save_checkpoint(model, path, {'step': 2})
            load_checkpoint(restored, path, strict=True)
            self.assertTrue(torch.equal(model.weight, restored.weight))

    def test_zero_worker_loader_and_single_engine_cli(self):
        args = get_args(['--config', 'x.yaml', '--model_dir', 'out', '--train_data', 'data'])
        self.assertEqual(list(make_loader([1, 2], args, 7)), [1, 2])
        self.assertFalse(hasattr(args, 'train_engine'))

    def test_training_entrypoint_with_current_model_interface(self):
        for dtype_name in ['fp32', 'bf16', 'fp16']:
            with self.subTest(dtype=dtype_name):
                model = TinyLoss()
                model.acoustic_tokenizer = torch.nn.Linear(1, 1)
                model.diffusion_head = torch.nn.Linear(1, 1)
                model.tokenizer = object()
                config = {'temporal_model': 'stub', 'vibevoice_path': 'stub', 'diffusion_head_type': 'llm',
                          'diffusion_head': 'stub', 'dataset_conf': {}, 'optim_conf': {'lr': .01},
                          'scheduler_conf': {'warmup_steps': 4}, 'accum_grad': 2,
                          'max_epoch': 1, 'max_steps': 2, 'dtype': dtype_name}
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = root / 'config.yaml'; path.write_text(yaml.safe_dump(config))
                    with patch('twinlakes.models.rq_transformer.LMModel.from_audio_text_pretrained', return_value=model), patch('twinlakes.dataset.dataset.Dataset', return_value=[batch()] * 3):
                        train_main(['--config', str(path), '--model_dir', str(root / 'model'), '--train_data', 'stub', '--device', 'cpu'])
                    saved = torch.load(root / 'model/0_2.pt', weights_only=False)
                    self.assertEqual(saved['infos']['step'], 2)
                    self.assertIn('optimizer', saved)
                    self.assertEqual(len(saved['optimizer']['param_groups'][0]['params']), 3)
                    self.assertTrue(all(not p.requires_grad for p in model.acoustic_tokenizer.parameters()))
                    expected = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[dtype_name]
                    self.assertTrue(all(p.dtype == expected for p in model.parameters()))
                    self.assertTrue(all(value.dtype == expected for value in saved['model'].values() if value.is_floating_point()))
                    self.assertEqual(model.seen_dtypes['wavs'], expected)
        

    def test_uneven_rank_input_stops_before_next_forward(self):
        calls = []
        def all_reduce(flag, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                flag.zero_()
        with patch('twinlakes.utils.executor.dist.is_initialized', return_value=True), patch('twinlakes.utils.executor.dist.all_reduce', side_effect=all_reduce):
            self.assertEqual(list(synchronized_batches([1, 2, 3], 'cpu')), [1])
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
