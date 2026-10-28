"""Binary paired-latent TAR protocol shared by packing and training."""
import io
import json
import math
from pathlib import Path
import re

import torch

from .audio_shard import EMILIA_ID

LATENT_SHARD_FORMAT = 'emilia_latent_pair_v1'
CODEC_FIELDS = ('encoder_sha256', 'implementation_sha256', 'sample_rate', 'channels',
                'dimension', 'hop_length', 'dtype', 'latent_mode', 'scaled',
                'speech_scaling_factor', 'speech_bias_factor', 'normalization')
DTYPES = {'torch.bfloat16': torch.bfloat16, 'torch.float16': torch.float16, 'torch.float32': torch.float32}


def codec_signature(metadata):
    result = {key: metadata[key] for key in CODEC_FIELDS}
    if (result['scaled'] is not False or result['dimension'] != 64 or result['hop_length'] != 3200
            or result['sample_rate'] != 24000 or result['channels'] != 1
            or result['dtype'] not in DTYPES or 'encoder mean' not in result['latent_mode']
            or not re.fullmatch('[0-9a-f]{64}', result['encoder_sha256'])):
        raise ValueError('Expected raw TwinLakes acoustic encoder mean, 24kHz/3200/64 dimensions')
    if not all(math.isfinite(result[key]) for key in ('speech_scaling_factor', 'speech_bias_factor')):
        raise ValueError('Nonfinite codec affine parameters')
    if result['speech_scaling_factor'] <= 0:
        raise ValueError('Codec scale must be positive')
    return result


def load_codec_metadata(path):
    metadata = json.loads(Path(path).read_text())
    codec_signature(metadata)
    return metadata


def resolve_latent_path(value, language, root):
    """Accept an Emilia ID, relative wav path, or absolute wav/latent path."""
    if language not in ('en', 'zh') or not isinstance(value, str) or not value:
        raise ValueError('Invalid Emilia language or audio path/ID')
    if EMILIA_ID.fullmatch(value):
        speaker = value.rsplit('_W', 1)[0]
        relative = Path(speaker.split('_S', 1)[0]) / speaker / 'mp3' / (value + '.latent')
    else:
        path = Path(value)
        if '..' in path.parts:
            raise ValueError('Parent traversal in latent path')
        parts = list(path.parts)
        if path.is_absolute():
            offsets = [i for i, part in enumerate(parts) if re.fullmatch(r'(EN|ZH)_B[0-9]+', part)]
            if len(offsets) != 1:
                raise ValueError(f'Cannot derive Emilia relative path: {value}')
            parts = parts[offsets[0]:]
        elif parts and parts[0] in ('EN', 'ZH'):
            if parts.pop(0) != language.upper():
                raise ValueError('Latent language prefix mismatch')
        relative = Path(*parts).with_suffix('.latent')
    if not relative.parts[0].startswith(language.upper() + '_B'):
        raise ValueError('Latent path/ID language mismatch')
    if not EMILIA_ID.fullmatch(relative.stem) or len(relative.parts) != 4:
        raise ValueError(f'Invalid Emilia latent path: {value}')
    return Path(root) / language.upper() / relative


def load_latent_bytes(payload, codec):
    tensor = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
            or tensor.shape[0] < 1 or tensor.shape[1] != codec['dimension']
            or tensor.dtype != DTYPES[codec['dtype']] or not torch.isfinite(tensor).all()):
        raise ValueError('Invalid latent tensor shape/dtype/values')
    return tensor.contiguous()


def decode_latent_pair(sample, expected_codec=None):
    row = json.loads(sample['json'])
    key = sample['key']
    if row.get('shard_format') != LATENT_SHARD_FORMAT or row.get('id') != key:
        raise ValueError('Latent shard format or key mismatch')
    codec = codec_signature(row['latent_codec'])
    if expected_codec is not None and codec != codec_signature(expected_codec):
        raise ValueError(f'Latent codec differs from configured metadata: {key}')
    if row['latent_members'] != {'target': key + '.latent', 'prompt': key + '.prompt'}:
        raise ValueError('Latent member names differ from metadata')
    target = load_latent_bytes(sample['latent'], codec)
    prompt = load_latent_bytes(sample['prompt'], codec)
    if row['latent_shapes'] != {'target': list(target.shape), 'prompt': list(prompt.shape)}:
        raise ValueError('Latent member shape differs from metadata')
    return dict(key=key, text=row['text'], prompt_text=row['prompt_text'],
                wav=target, prompt_wav=prompt, is_latent=True,
                sample_rate=codec['sample_rate'],
                latent_codec=codec)
