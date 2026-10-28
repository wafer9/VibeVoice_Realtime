# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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

"""Local logging and runtime configuration for native PyTorch training."""
import logging
import os
import torch

from twinlakes.utils.common import tensor_to_scalar


def get_torch_dtype(name):
    dtypes = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    if name not in dtypes:
        raise ValueError('dtype must be fp32, bf16, or fp16')
    return dtypes[name]


def check_modify_and_save_config(args, configs):
    configs.pop('train_engine', None)
    configs['dtype'] = args.dtype or configs.get('dtype', 'fp32')
    get_torch_dtype(configs['dtype'])
    if int(configs.get('accum_grad', 1)) < 1:
        raise ValueError('accum_grad must be positive')
    configs['model_dir'] = args.model_dir
    return configs


def log_per_step(writer, info_dict, timer=None):
    if int(os.environ.get('RANK', 0)) != 0:
        return
    step, tag = info_dict['step'], info_dict['tag']
    if tag == 'TRAIN' and step % info_dict.get('log_interval', 10):
        return
    values = {key: tensor_to_scalar(value) for key, value in info_dict['loss_dict'].items() if value is not None}
    if writer is not None:
        for key, value in values.items():
            writer.add_scalar(f'{tag.lower()}/{key}', value, step)
        if tag == 'TRAIN':
            writer.add_scalar('train/lr', info_dict['lrs'][0], step)
            writer.add_scalar('train/grad_norm', info_dict['grad_norm'], step)
    rate = timer.steps_per_second(step) if timer is not None else 0
    details = ' '.join(f'{key}={value:.5f}' for key, value in values.items())
    logging.info('%s epoch=%s step=%s %s steps/sec=%.2f', tag, info_dict.get('epoch', 0), step, details, rate)
