# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate paired-reference TTS with the current LLM-form diffusion head.

One process handles all rows; torchrun processes handle disjoint row indices.
"""
import argparse
import logging
import math
import os
from pathlib import Path

import torch
import yaml

from twinlakes.dataset.audio_shard import resolve_audio_path
from twinlakes.utils.checkpoint import load_checkpoint
from twinlakes.utils.train_utils import get_torch_dtype


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'test_data', 'checkpoint', 'result_dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--gpu', type=int, default=-1, help='-1 = CPU; torchrun uses LOCAL_RANK')
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], help='Override config dtype')
    parser.add_argument('--cfg_scale', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--en-root', default='/nfs-speech-cfs/ASR/opensource/Emilia/EN')
    parser.add_argument('--zh-root', default='/nfs-speech-cfs/ASR/opensource/Emilia/ZH')
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv=None):
    args = get_args(argv)
    # Lazy imports keep --help usable without loading the model/audio stack.
    import json
    import torchaudio
    from twinlakes.models.rq_transformer import LMModel
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    gpu = int(os.environ.get('LOCAL_RANK', args.gpu)) if world > 1 else args.gpu
    device = torch.device('cpu' if gpu < 0 else f'cuda:{gpu}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    configs = yaml.safe_load(Path(args.config).read_text())
    configs['dtype'] = args.dtype or configs.get('dtype', 'fp32')
    dtype = get_torch_dtype(configs['dtype'])
    model = LMModel.from_audio_text_pretrained(configs)
    load_checkpoint(model, args.checkpoint)
    model.to(device=device, dtype=dtype).eval()
    output = Path(args.result_dir)
    output.mkdir(parents=True, exist_ok=True)
    roots = {'en': args.en_root, 'zh': args.zh_root}
    with Path(args.test_data).open() as source:
        for index, line in enumerate(source):
            if index % world != rank or not line.strip():
                continue
            row = json.loads(line)
            key = row.get('id', row.get('key'))
            if not key or Path(key).name != key:
                raise ValueError('Each row needs a safe id/key')
            language = row.get('language', key.split('_')[0].lower())
            waveform, sample_rate = torchaudio.load(str(resolve_audio_path(row['prompt_wav'], language, roots)))
            waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 24000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)
            frames = math.ceil(waveform.shape[-1] / 3200)
            prefix = ' Voice input:\n<|vision_start|>%s<|vision_end|>\n Text input:\n%s\n Speech output:\n<|vision_start|>' % ('<|vision_pad|>' * frames, row['text'])
            ids = torch.tensor(model.tokenizer.encode(prefix), device=device).unsqueeze(0)
            starts = (ids[0] == model.tokenizer.speech_start_id).nonzero().flatten()
            ends = (ids[0] == model.tokenizer.speech_end_id).nonzero().flatten()
            positions = torch.stack([starts[0], ends[0], starts[1]])
            audio = model.generate_cfg_locdit(keys=[key], prompt_wavs=waveform.unsqueeze(-1).to(device=device, dtype=dtype),
                                             prompt_wavs_lengths=torch.tensor([waveform.shape[-1]], device=device),
                                             input_ids=ids, audio_pos=[positions], cfg_scale=args.cfg_scale)
            torchaudio.save(str(output / f'{key}.wav'), audio.float().cpu(), 24000)
            logging.info('Generated %s', key)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
