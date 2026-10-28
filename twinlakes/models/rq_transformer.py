# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn

from transformers import PreTrainedModel
from typing import Optional, Dict, List
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import copy

from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
from vibevoice.modular.modular_vibevoice_diffusion_head import TimestepEmbedder
from twinlakes.utils.train_utils import get_torch_dtype


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class LMModel(nn.Module):
    """Current user model: LLM, acoustic connector, and diffusion head."""

    def __init__(
        self,
        config: Optional[VibeVoiceConfig] = None,
        language_model: Optional[PreTrainedModel] = None,
        acoustic_tokenizer: Optional[PreTrainedModel] = None,
        acoustic_connector: Optional[SpeechConnector] = None,
        diffusion_head: Optional[VibeVoiceDiffusionHead] = None,
        noise_scheduler: Optional[DPMSolverMultistepScheduler] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        speech_scaling_factor: Optional[torch.tensor] = None,
        speech_bias_factor: Optional[torch.tensor] = None,
        # diffusion_head_proj: Optional[torch.nn.Linear] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.lm = language_model
        self.acoustic_tokenizer = acoustic_tokenizer
        self.acoustic_connector = acoustic_connector

        self.diffusion_head = diffusion_head
        # self.diffusion_head_proj = diffusion_head_proj
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer

        self.register_buffer('speech_scaling_factor', speech_scaling_factor)
        self.register_buffer('speech_bias_factor', speech_bias_factor)

        self.lm_head = torch.nn.Linear(self.lm.config.hidden_size, 2)
        latent_size = self.diffusion_head.config.latent_size
        self.audio_bos = nn.Parameter(torch.zeros(1, 1, latent_size))
        self.uncondition_mlp = nn.Sequential(
            nn.Linear(latent_size, 256), nn.SiLU(),
            nn.Linear(256, self.lm.config.hidden_size))



    @classmethod
    def from_audio_text_pretrained(
        self,
        configs: Optional[Dict] = {},
    ):
        
        dtype = get_torch_dtype(configs.get('dtype', 'fp32'))
        lm_path = configs['temporal_model']
        lm_config = AutoConfig.from_pretrained(lm_path, trust_remote_code=True)
        language_model = AutoModelForCausalLM.from_pretrained(lm_path, trust_remote_code=True, dtype=dtype)

        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(lm_path)

        vibevoice = VibeVoiceForConditionalGenerationInference.from_pretrained(configs['vibevoice_path'], dtype=dtype)
        acoustic_tokenizer = vibevoice.model.acoustic_tokenizer
        acoustic_connector = SpeechConnector(acoustic_tokenizer.config.vae_dim, lm_config.hidden_size)


        diffusion_head_config = copy.deepcopy(vibevoice.config.diffusion_head_config)
        diffusion_head_config.hidden_size = lm_config.hidden_size
        diffusion_head = AutoModel.from_config(diffusion_head_config)
        # diffusion_head_proj = torch.nn.Linear(lm_config.hidden_size, diffusion_head_config.hidden_size)

        # diffusion_head = vibevoice.prediction_head
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=vibevoice.config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=vibevoice.config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=vibevoice.config.diffusion_head_config.prediction_type
        )

        model = self(
                    config=vibevoice.config,
                    language_model=language_model,
                    acoustic_tokenizer=acoustic_tokenizer,
                    acoustic_connector=acoustic_connector,
                    diffusion_head=diffusion_head,
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    speech_scaling_factor=vibevoice.speech_scaling_factor,
                    speech_bias_factor=vibevoice.speech_bias_factor,
                    # diffusion_head_proj=diffusion_head_proj,
                    )
        return model.to(dtype=dtype)
    


    def forward(self,
                keys: List[str],
                input_ids: torch.Tensor,
                labels: torch.Tensor,
                prompt_wavs: torch.Tensor,
                wavs: torch.Tensor,
                prompt_wavs_lengths: torch.Tensor,
                wavs_lengths: torch.Tensor,
                audio_pos: List[torch.Tensor],
                ):

        B, T  = input_ids.shape
        mask = (0 <= input_ids) & (input_ids < self.lm.vocab_size)
        x = self.lm.get_input_embeddings()(torch.masked_fill(input_ids, ~mask, self.tokenizer.eos_id))

        # 自动识别输入：last dim == vae_dim → 已是预存 latent(新 shard，长度按帧数)；
        # 否则是原始音频 [B, L, 1]，在线 encode(长度按采样数/3200)。
        vae_dim = self.config.acoustic_vae_dim

        if prompt_wavs.shape[-1] == vae_dim:
            prompt_speech_masks = ~make_pad_mask(prompt_wavs_lengths.to(dtype=torch.int64))
            prompt_speech_features = prompt_wavs
        else:
            prompt_speech_masks = ~make_pad_mask(torch.ceil(prompt_wavs_lengths/3200).to(dtype=torch.int64))
            prompt_speech_features = self.acoustic_tokenizer.encode(prompt_wavs.transpose(1,2)).mean
        prompt_speech_features = (prompt_speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        prompt_speech_connect_features = self.acoustic_connector(prompt_speech_features)


        if wavs.shape[-1] == vae_dim:
            speech_masks = ~make_pad_mask(wavs_lengths.to(dtype=torch.int64))
            speech_features = wavs
        else:
            speech_masks = ~make_pad_mask(torch.ceil(wavs_lengths/3200).to(dtype=torch.int64))
            speech_features = self.acoustic_tokenizer.encode(wavs.transpose(1,2)).mean
        speech_features = (speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        speech_connect_features = self.acoustic_connector(speech_features)


        prompt_wav_pad_mask = torch.zeros_like(input_ids)
        wav_pad_mask = torch.zeros_like(input_ids)
        wav_loss_mask = torch.zeros_like(input_ids)

        for idx in range(len(audio_pos)):
            if audio_pos[idx].shape[0] == 4:
                p_s, p_e, s, e = audio_pos[idx].tolist()
                prompt_wav_pad_mask[idx, p_s+1:p_e] = 1
                wav_pad_mask[idx, s+1:e] = 1
                wav_loss_mask[idx, s:e-1] = 1 # 包括
            elif audio_pos[idx].shape[0] == 2:
                s, e = audio_pos[idx].tolist()
                wav_pad_mask[idx, s+1:e] = 1
                wav_loss_mask[idx, s:e-1] = 1
                prompt_speech_masks[idx] = 0

        
        prompt_wav_pad_mask = prompt_wav_pad_mask.bool()
        wav_pad_mask = wav_pad_mask.bool()
        wav_loss_mask = wav_loss_mask.bool()


        x[prompt_wav_pad_mask] = prompt_speech_connect_features[prompt_speech_masks].to(dtype=x.dtype)
        x[wav_pad_mask] = speech_connect_features[speech_masks].to(dtype=x.dtype)

        # cond 计算text loss
        cond_outputs = self.lm(inputs_embeds=x,
                            past_key_values=None,
                            attention_mask=None,
                            labels=None,
                            use_cache=None,
                            output_attentions=None,
                            output_hidden_states=True,
                            return_dict=True)

        hidden_states = cond_outputs.hidden_states[-1]
        logits = self.lm_head(hidden_states)

        loss_text = None
        if labels is not None:
            labels[labels==self.tokenizer.speech_diffusion_id] = 0
            labels[labels==self.tokenizer.eos_id] = 1
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss_text = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        # --- Diffusion Loss Calculation ---
        ddpm_batch_mul = 1
        diffusion_loss = None

        # This block is executed only if we are in a context that involves speech.
        if wav_loss_mask.sum().item() > 0:
            condition_features = hidden_states[wav_loss_mask]
            speech_features_2d = speech_features[speech_masks]
            speech_len, latent_size = speech_features_2d.shape
            
            noise = torch.randn(
                (speech_len * ddpm_batch_mul, latent_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            timesteps = torch.multinomial(
                torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
                speech_len * ddpm_batch_mul,
                replacement=True,
            ).to(hidden_states.device)

            speech_features_repeated = speech_features_2d.repeat_interleave(ddpm_batch_mul, dim=0)
            condition_features_repeated = condition_features.repeat_interleave(ddpm_batch_mul, dim=0)
            # Shift within each utterance, in normalized latent space; text positions are irrelevant.
            previous_speech = torch.cat(
                [self.audio_bos.expand(B, -1, -1), speech_features[:, :-1]], dim=1)
            uncondition_features = self.uncondition_mlp(previous_speech[speech_masks])
            uncondition_features = uncondition_features.repeat_interleave(ddpm_batch_mul, dim=0)

            noisy_speech_features = self.noise_scheduler.add_noise(
                speech_features_repeated, noise, timesteps
            )
            model_output = self.diffusion_head(
                noisy_speech_features, 
                timesteps.type_as(x), 
                condition_features_repeated
            )
            uncondition_output = self.diffusion_head(
                noisy_speech_features, timesteps.type_as(x), uncondition_features)
            target_for_loss = self.noise_scheduler.get_velocity(
                speech_features_repeated, noise, timesteps
            )

            condition_loss = torch.nn.functional.mse_loss(model_output.float(), target_for_loss.float())
            uncondition_loss = torch.nn.functional.mse_loss(uncondition_output.float(), target_for_loss.float())
            diffusion_loss = 0.8 * condition_loss + 0.2 * uncondition_loss


        else:
            # Dummy loss for DDP to work when there are no speech samples in a batch,
            # but we are in a speech context.
            diffusion_loss = sum(p.sum() for p in self.diffusion_head.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.acoustic_connector.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.uncondition_mlp.parameters()) * 0.0
            diffusion_loss += self.audio_bos.sum() * 0.0
            condition_loss = uncondition_loss = diffusion_loss
        # --- End Diffusion Loss Calculation ---

        loss = loss_text + diffusion_loss
        return {"loss": loss, "loss_text": loss_text, "diffusion_loss": diffusion_loss,
                "condition_loss": condition_loss, "uncondition_loss": uncondition_loss}


    @torch.no_grad()
    def generate_cfg(self,
                keys, prompt_wavs, prompt_wavs_lengths, input_ids, audio_pos
                ):
        B, T  = input_ids.shape
        device = input_ids.device
        mask = (0 <= input_ids) & (input_ids < self.lm.vocab_size)
        x = self.lm.get_input_embeddings()(torch.masked_fill(input_ids, ~mask, self.tokenizer.eos_id))
        negative_x = x[:, -4:, :]

        prompt_speech_masks = ~make_pad_mask(torch.ceil(prompt_wavs_lengths/3200).to(dtype=torch.int64))
        prompt_speech_features = self.acoustic_tokenizer.encode(prompt_wavs.transpose(1,2)).mean
        prompt_speech_features = (prompt_speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        prompt_speech_connect_features = self.acoustic_connector(prompt_speech_features)

        prompt_audio_pad_masks = torch.zeros_like(input_ids)
        for idx in range(len(audio_pos)):
            s, e = audio_pos[idx].tolist()
            prompt_audio_pad_masks[idx, s+1:e] = 1

        prompt_audio_pad_masks = prompt_audio_pad_masks.bool()
        x[prompt_audio_pad_masks] = prompt_speech_connect_features[prompt_speech_masks] # 拼接 prompt wav

        audios = []
        past_key_values = None
        negative_past_key_values = None
        diffusion_indices = torch.arange(B, device=device)
        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()

        n = 0
        while True:
            outputs = self.lm(inputs_embeds=x,
                        past_key_values=past_key_values,
                        attention_mask=None,
                        labels=None,
                        use_cache=None,
                        output_attentions=None,
                        output_hidden_states=True,
                        return_dict=True)
            
            negative_outputs = self.lm(inputs_embeds=negative_x,
                        past_key_values=negative_past_key_values,
                        attention_mask=None,
                        labels=None,
                        use_cache=None,
                        output_attentions=None,
                        output_hidden_states=True,
                        return_dict=True)
            
            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1][:, -1:, :]

            logits = self.lm_head(hidden_states)
            next_tokens = torch.argmax(logits, dim=-1)
            if next_tokens[0] == 1:
                break

            negative_past_key_values = negative_outputs.past_key_values
            negative_hidden_states = negative_outputs.hidden_states[-1][:, -1:, :]

            
            speech_latent = self.sample_speech_tokens_cfg(hidden_states[0], negative_hidden_states[0], cfg_scale=1.4).unsqueeze(1)
            # speech_latent = self.sample_speech_tokens(hidden_states).unsqueeze(1)

            scaled_latent = speech_latent / self.speech_scaling_factor.to(speech_latent.device) - self.speech_bias_factor.to(speech_latent.device)

            audio_chunk = self.acoustic_tokenizer.decode(
                        scaled_latent.to(self.acoustic_tokenizer.device),
                        cache=acoustic_cache,  # Use acoustic-specific cache
                        sample_indices=diffusion_indices.to(self.acoustic_tokenizer.device),
                        use_cache=True,
                        debug=False
                    )
            audios.append(audio_chunk)
            
            semantic_features = self.semantic_tokenizer.encode(
                audio_chunk,
                cache=semantic_cache,  # Use semantic-specific cache
                sample_indices=diffusion_indices,
                use_cache=True,
                debug=False
            ).mean # semantic tokenizer has no VAE.

            acoustic_embed = self.acoustic_connector(speech_latent)
            semantic_embed = self.semantic_connector(semantic_features)
            
            x = acoustic_embed + semantic_embed
            negative_x = acoustic_embed + semantic_embed


        y = torch.cat(audios, dim=2)[0]
        return y.float().detach().cpu()


    @torch.no_grad()
    def sample_speech_tokens(self, condition, cfg_scale=3.0):
        ddpm_inference_steps = 10
        self.noise_scheduler.set_timesteps(ddpm_inference_steps)
        # condition = torch.cat([condition, neg_condition], dim=0).to(self.diffusion_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.noise_scheduler.timesteps:
            eps = self.diffusion_head(speech, t.repeat(speech.shape[0]).to(speech), condition=condition)
            speech = self.noise_scheduler.step(eps, t, speech).prev_sample
        return speech


    @torch.no_grad()
    def sample_speech_tokens_cfg(self, condition, neg_condition, cfg_scale=1.0):
        ddpm_inference_steps = 10
        self.noise_scheduler.set_timesteps(ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.diffusion_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.diffusion_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.noise_scheduler.step(eps, t, speech).prev_sample

        return speech[: len(speech) // 2]
