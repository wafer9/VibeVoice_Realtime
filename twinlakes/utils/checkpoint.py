# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

"""Native PyTorch checkpoints, with support for previous weight-only files."""
import datetime
import logging
from pathlib import Path

import torch
import yaml


def load_checkpoint(model, path, *, optimizer=None, scheduler=None, scaler=None, strict=False):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    full_state = isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict)
    weights = checkpoint['model'] if full_state else checkpoint
    result = model.load_state_dict(weights, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        logging.warning('Checkpoint keys: %d missing, %d unexpected', len(result.missing_keys), len(result.unexpected_keys))
    for name, target in [('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)]:
        if target is not None and full_state and checkpoint.get(name) is not None:
            target.load_state_dict(checkpoint[name])
    sidecar = Path(path).with_suffix('.yaml')
    infos = checkpoint.get('infos', {}) if full_state else {}
    if sidecar.is_file():
        infos = yaml.safe_load(sidecar.read_text()) or infos
    if optimizer is not None and not full_state:
        logging.warning('Weight-only checkpoint: optimizer state starts fresh')
    return infos


def save_state_dict_and_infos(state_dict, path, infos=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    infos = {**(infos or {}), 'save_time': datetime.datetime.now().isoformat(timespec='seconds')}
    temporary = path.with_name(path.name + '.partial')
    torch.save(state_dict, temporary)
    temporary.replace(path)
    path.with_suffix('.yaml').write_text(yaml.safe_dump(infos, allow_unicode=True))


def save_checkpoint(model, path, infos=None, *, optimizer=None, scheduler=None, scaler=None):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    state = model.state_dict()
    if optimizer is not None:
        state = {'model': state, 'optimizer': optimizer.state_dict(), 'infos': dict(infos or {}),
                 'scheduler': scheduler.state_dict() if scheduler is not None else None,
                 'scaler': scaler.state_dict() if scaler is not None else None}
    save_state_dict_and_infos(state, path, infos)
