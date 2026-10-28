"""Verify configured parameter/input dtypes with real tiny Qwen/VibeVoice modules."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from twinlakes.models.rq_transformer import LMModel
from vibevoice.modular.configuration_vibevoice import VibeVoiceDiffusionHeadConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler


class TinyCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv1d(1, 4, kernel_size=1, stride=3200)

    def encode(self, waveform):
        return SimpleNamespace(mean=self.encoder(waveform).transpose(1, 2))


def make_model():
    lm_config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
    head_config = VibeVoiceDiffusionHeadConfig(hidden_size=16, latent_size=4,
                                             head_layers=1, ddpm_num_steps=32)
    model = LMModel(config=SimpleNamespace(acoustic_vae_dim=4, diffusion_head_config=head_config),
                    language_model=Qwen3ForCausalLM(lm_config), acoustic_tokenizer=TinyCodec(),
                    acoustic_connector=SpeechConnector(4, 16),
                    diffusion_head=VibeVoiceDiffusionHead(head_config),
                    noise_scheduler=DPMSolverMultistepScheduler(num_train_timesteps=32,
                                                               beta_schedule='cosine', prediction_type='v_prediction'),
                    tokenizer=SimpleNamespace(eos_id=1, speech_diffusion_id=12),
                    speech_scaling_factor=torch.tensor(.2), speech_bias_factor=torch.tensor(-.05))
    model.float()
    model.acoustic_tokenizer.requires_grad_(False)
    return model


def make_batch(raw):
    ids = torch.tensor([[0, 10, 12, 12, 11, 0, 10, 12, 12, 12, 12, 1],
                        [0, 10, 12, 11, 0, 10, 12, 12, 12, 1, -100, -100]])
    labels = torch.full_like(ids, -100)
    labels[0, 7:11], labels[0, 11] = 12, 1
    labels[1, 6:9], labels[1, 9] = 12, 1
    factor = 3200 if raw else 1
    return {'keys': ['a', 'b'], 'input_ids': ids, 'labels': labels,
            'prompt_wavs': torch.randn(2, 2 * factor, 1 if raw else 4),
            'wavs': torch.randn(2, 4 * factor, 1 if raw else 4),
            'prompt_wavs_lengths': torch.tensor([2, 1]) * factor,
            'wavs_lengths': torch.tensor([4, 3]) * factor,
            'audio_pos': [torch.tensor([1, 4, 6, 11]), torch.tensor([1, 3, 5, 9])]}


class ModelAMPTests(unittest.TestCase):
    def test_real_forward_backward_for_audio_and_padded_latents(self):
        torch.set_num_threads(2)
        for raw in [False, True]:
            for dtype in [torch.float32, torch.bfloat16, torch.float16]:
                with self.subTest(raw_audio=raw, dtype=dtype):
                    torch.manual_seed(8)
                    model = make_model().to(dtype=dtype)
                    self.assertTrue(all(p.dtype == dtype for p in model.parameters()))
                    self.assertTrue(all(b.dtype == dtype for b in model.buffers() if b.is_floating_point()))
                    batch = make_batch(raw)
                    batch = {key: value.to(dtype=dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value for key, value in batch.items()}
                    connector_dtypes, head_shapes = [], []
                    model.acoustic_connector.register_forward_hook(lambda module, args, out: connector_dtypes.append(out.dtype))
                    model.diffusion_head.register_forward_pre_hook(lambda module, args: head_shapes.append(tuple(args[0].shape)))
                    with torch.autocast('cpu', dtype=dtype) if dtype != torch.float32 else nullcontext():
                        result = model(**batch)
                    self.assertTrue(torch.isfinite(result['loss']))
                    self.assertEqual(connector_dtypes, [dtype] * 2)
                    # Two targets have 4 and 3 valid frames, excluding padding.
                    self.assertEqual(head_shapes, [(7, 4), (7, 4)])
                    torch.testing.assert_close(result['diffusion_loss'],
                                               .8 * result['condition_loss'] + .2 * result['uncondition_loss'])
                    result['loss'].backward()
                    gradient = model.acoustic_connector.fc1.weight.grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
                    self.assertIsNotNone(model.lm.get_input_embeddings().weight.grad)
                    self.assertTrue(all(p.grad is None for p in model.acoustic_tokenizer.parameters()))

    def test_uncondition_history_weighting_and_gradient_isolation(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        model = make_model()
        # A trained head has nonzero output/modulation weights; exercise both gradient paths.
        torch.nn.init.normal_(model.diffusion_head.final_layer.linear.weight, std=.05)
        torch.nn.init.normal_(model.diffusion_head.final_layer.adaLN_modulation[-1].weight, std=.05)
        with torch.no_grad():
            model.audio_bos.fill_(.4)
        sample = make_batch(raw=False)
        previous_inputs, head_calls = [], []
        model.uncondition_mlp.register_forward_pre_hook(lambda module, args: previous_inputs.append(args[0]))
        model.diffusion_head.register_forward_hook(lambda module, args, out: head_calls.append((args, out)))
        with patch.object(model.lm, 'forward', wraps=model.lm.forward) as llm, \
             patch.object(model.noise_scheduler, 'get_velocity', wraps=model.noise_scheduler.get_velocity) as velocity:
            result = model(**sample)
            self.assertEqual(llm.call_count, 1)
            target_args = velocity.call_args.args
        features = (sample['wavs'] + model.speech_bias_factor) * model.speech_scaling_factor
        expected = torch.cat([model.audio_bos[0], features[0, :3], model.audio_bos[0], features[1, :2]])
        torch.testing.assert_close(previous_inputs[0], expected)
        self.assertEqual(tuple(previous_inputs[0].shape), (7, 4))
        # Both branches see exactly the same noisy targets and times, excluding padding.
        self.assertEqual(len(head_calls), 2)
        self.assertIs(head_calls[0][0][0], head_calls[1][0][0])
        torch.testing.assert_close(head_calls[0][0][1], head_calls[1][0][1])
        self.assertFalse(torch.equal(head_calls[0][0][2], head_calls[1][0][2]))
        target = model.noise_scheduler.get_velocity(*target_args)
        for name, (_, output) in zip(('condition_loss', 'uncondition_loss'), head_calls):
            torch.testing.assert_close(result[name], torch.nn.functional.mse_loss(output.float(), target.float()))
        torch.testing.assert_close(result['loss'], result['loss_text'] + .8*result['condition_loss'] + .2*result['uncondition_loss'])

        result['condition_loss'].backward(retain_graph=True)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.lm.parameters()))
        self.assertTrue(all(p.grad is None for p in model.uncondition_mlp.parameters()))
        self.assertIsNone(model.audio_bos.grad)
        model.zero_grad(set_to_none=True)
        result['uncondition_loss'].backward()
        self.assertTrue(all(p.grad is None for p in model.lm.parameters()))
        for parameter in [model.audio_bos, *model.uncondition_mlp.parameters()]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.acoustic_tokenizer.parameters()))

    def test_no_target_frames_keeps_uncondition_parameters_in_graph(self):
        model = make_model()
        sample = dict(keys=['empty'], input_ids=torch.tensor([[0, 10, 1]]),
                      labels=torch.tensor([[-100, -100, 1]]),
                      prompt_wavs=torch.zeros(1, 1, 4), wavs=torch.zeros(1, 0, 4),
                      prompt_wavs_lengths=torch.tensor([1]), wavs_lengths=torch.tensor([0]),
                      audio_pos=[torch.tensor([1, 2])])
        result = model(**sample)
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertEqual(result['condition_loss'].item(), 0)
        self.assertEqual(result['uncondition_loss'].item(), 0)
        result['loss'].backward()
        for parameter in [model.audio_bos, *model.uncondition_mlp.parameters()]:
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
