# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
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

import io
import json
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import math
import base64
import numpy as np

try:
    torchaudio.utils.sox_utils.set_buffer_size(16500)
except (AttributeError, OSError, RuntimeError):
    pass

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
from vibevoice.processor.vibevoice_processor import AudioNormalizer
audio_normalizer = AudioNormalizer()
SAMPLE_RATE=24000


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    row = json.loads(elem['line'])
    row.setdefault('key', row.get('id'))
    row['file_name'] = elem['file_name']
    return row


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def decode_wav_raw(sample, resample=24000):
    from .audio_shard import resolve_audio_path
    language = sample.get('language', sample['key'].split('_')[0].lower())
    roots = {'en': '/nfs-speech-cfs/ASR/opensource/Emilia/EN',
             'zh': '/nfs-speech-cfs/ASR/opensource/Emilia/ZH'}
    row = dict(sample)
    for name in ['wav', 'prompt_wav']:
        path = resolve_audio_path(sample[name], language, roots)
        waveform, rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if rate != resample:
            waveform = torchaudio.functional.resample(waveform, rate, resample)
        row[name] = waveform
    row.setdefault('prompt_text', '')
    return row


def decode_wav_single(sample):
    from .audio_shard import decode_audio_pair
    if 'json' in sample:
        obj = decode_audio_pair(sample)
        payloads = [obj['target_audio_bytes'], obj['prompt_audio_bytes']]
    else:
        obj = json.loads(sample['text'])
        payloads = [base64.b64decode(obj[name]) for name in ['wav', 'prompt_wav']]
    decoded = []
    for payload in payloads:
        waveform, rate = torchaudio.load(io.BytesIO(payload))
        waveform = waveform.mean(dim=0, keepdim=True)
        if rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
        decoded.append(waveform)
    return dict(key=obj['id'], wav=decoded[0], prompt_wav=decoded[1],
                text=obj['text'], prompt_text=obj.get('prompt_text', ''),
                sample_rate=SAMPLE_RATE)


def decode_latent_single(sample, max_per_line=None, expected_codec=None):
    """新 latent shard：每个 .text 是 json，含 base64(np.save, fp16) 的 VAE latent，
    直接解码成张量,免去在线 encode。

    产出的 'wav'/'prompt_wav' 是 **latent** 张量 [T, vae_dim](不是音频),并置 is_latent=True，
    下游 tokenize/filter/padding/forward 会据此走 latent 分支(长度按帧数而非采样数)。
    """
    if 'json' in sample:
        from .latent_shard import decode_latent_pair
        return decode_latent_pair(sample, expected_codec=expected_codec)
    if expected_codec is not None:
        raise ValueError('Legacy base64 latent has no codec provenance; cannot use expected_codec')
    assert 'key' in sample
    assert 'text' in sample
    obj = json.loads(sample['text'])

    def _lat(b64):
        arr = np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False)     # [T, vae_dim], fp16
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    tts_latent = _lat(obj['tts_latent'])                     # [T, 64]
    prompt_latent = _lat(obj['prompt_latent'])               # [Tp, 64]


    return dict(key=obj['id'], text=obj['text'],
                wav=tts_latent, prompt_wav=prompt_latent,
                is_latent=True, sample_rate=SAMPLE_RATE,
                prompt_text=obj['prompt_text'])


def sort_by_feats(sample):
    assert 'input_ids' in sample
    return sample['input_ids'].size(0)


def tokenize(sample, tokenizer, is_inference=False):
    speech_tok_compress_ratio = 3200
    is_latent = sample.get('is_latent', False)
    if is_latent:
        # latent 已是 [T, vae_dim]，vae token 数 = 帧数 = shape[0]
        prompt_vae_tok_len = sample['prompt_wav'].shape[0]
    else:
        prompt_vae_tok_len = math.ceil(sample['prompt_wav'].shape[1] / speech_tok_compress_ratio)
    prompt = " Voice input:\n<|vision_start|>%s<|vision_end|>\n Text input:\n%s\n Speech output:\n<|vision_start|>" \
                % ("<|vision_pad|>"*prompt_vae_tok_len, sample['text'])
    
    if is_latent:
        vae_tok_len = sample['wav'].shape[0] if sample['wav'].numel() > 0 else 0
    else:
        vae_tok_len = math.ceil(sample['wav'].shape[1] / speech_tok_compress_ratio)
    label = "<|vision_pad|>"*vae_tok_len + "<|endoftext|>"

    prompt, label = [prompt], [label]


    if is_inference:
        encoding = tokenizer(text=prompt,
                        text_target=label,
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True)
    else:
        encoding = tokenizer(
                            text=prompt,
                            text_pair=label,
                            add_special_tokens=True,
                            padding=True,  # truncation
                            return_tensors="pt",
                            return_token_type_ids=True,
                            return_attention_mask=True)
        token_type_ids = encoding["token_type_ids"]
        sample['labels'] = encoding["input_ids"].clone()
        sample["labels"][token_type_ids == 0] = -100
    prompt, label = prompt[0], label[0]

    sample['input_ids'] = encoding["input_ids"]

    start_pos = torch.where(encoding["input_ids"] == tokenizer.speech_start_id)
    end_pos = torch.where(encoding["input_ids"] == tokenizer.speech_end_id)
    eos_pos = torch.where(encoding["input_ids"] == tokenizer.eos_id)
    if start_pos[0].shape[0] == 2:
        audio_pos = torch.stack((start_pos[1][0], end_pos[1][0], start_pos[1][1], eos_pos[1][0]), dim=0)
    else:
        audio_pos = torch.stack((start_pos[1][0], eos_pos[1][0]), dim=0)

    if is_latent:
        # latent 直接透传 [T, vae_dim]，不做 audio_normalizer
        prompt_wav = sample['prompt_wav']
        wav = sample['wav']
    else:
        prompt_wav = torch.from_numpy(audio_normalizer(sample['prompt_wav'][0].numpy())).unsqueeze(0)
        wav = torch.from_numpy(audio_normalizer(sample['wav'][0].numpy())).unsqueeze(0) if sample['wav'].shape[1] > 0 else sample['wav']
    batch = {
        "key": sample['key'],
        "prompt": prompt,
        "label": label,
        "input_ids": sample['input_ids'][0],
        "label_ids": sample["labels"][0] if 'labels' in sample else torch.zeros([1, 0]),
        "prompt_wav": prompt_wav,
        "wav": wav,
        "audio_pos": audio_pos,
        "is_latent": is_latent,
        "text": sample['text'],
        "prompt_text": sample['prompt_text']
    }
    return batch


def filter(sample):
    flag = True
    if sample['wav'] is not None:
        if sample.get('is_latent', False):
            # latent [T, vae_dim]，按帧数过滤(7.5Hz)：30s≈225 帧，0.5s≈4 帧
            T = sample['wav'].shape[0]
            if T > math.ceil(30 * SAMPLE_RATE / 3200) or T < math.ceil(.5 * SAMPLE_RATE / 3200):
                flag = False
        else:
            if sample['wav'].shape[1] > 30*24000 or ( 0 < sample['wav'].shape[1] < 0.5*24000):
                flag = False

    if sample['text'] == sample['prompt_text']:
        flag = False
    return flag


def padding(data, ):

    sample = data
    assert isinstance(sample, list)
    input_ids_length = torch.tensor([x['input_ids'].size(0) for x in sample], dtype=torch.int32)
    order = torch.argsort(input_ids_length, descending=True)

    keys = [sample[i]['key'] for i in order]
    is_latent = sample[0].get('is_latent', False)
    if is_latent:
        # latent [T, vae_dim]：沿帧维(dim0)pad，长度=帧数；不 transpose
        prompt_wavs = [sample[i]['prompt_wav'] for i in order]
        wavs = [sample[i]['wav'] for i in order]
        prompt_wavs_lengths = torch.tensor([sample[i]['prompt_wav'].size(0) for i in order],
                                     dtype=torch.int32)
        wavs_lengths = torch.tensor([sample[i]['wav'].size(0) for i in order],
                                     dtype=torch.int32)
    else:
        prompt_wavs = [sample[i]['prompt_wav'].transpose(0,1) for i in order]
        wavs = [sample[i]['wav'].transpose(0,1) for i in order]
        prompt_wavs_lengths = torch.tensor([sample[i]['prompt_wav'].size(1) for i in order],
                                     dtype=torch.int32)
        wavs_lengths = torch.tensor([sample[i]['wav'].size(1) for i in order],
                                     dtype=torch.int32)

    input_ids = [sample[i]['input_ids'] for i in order]
    label_ids = [sample[i]['label_ids'] for i in order]

    padded_prompt_wavs = pad_sequence(prompt_wavs, batch_first=True, padding_value=0)
    padded_wavs = pad_sequence(wavs, batch_first=True, padding_value=0)
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    
    audio_pos = [sample[i]['audio_pos'] for i in order]

    batch = {
        "keys": keys,
        "prompt_wavs": padded_prompt_wavs,
        "wavs": padded_wavs,
        "input_ids": input_ids,
        "label_ids":label_ids,
        "prompt_wavs_lengths": prompt_wavs_lengths,
        "wavs_lengths": wavs_lengths,
        "audio_pos": audio_pos,
    }
    return batch


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'input_ids' in sample
        assert isinstance(sample['input_ids'], torch.Tensor)
        # new_sample_frames = sample['feat'].size(0)
        new_sample_frames = sample['input_ids'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False
