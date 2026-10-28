# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native PyTorch training, accumulation, mixed precision, and validation."""
from pathlib import Path

import torch
import torch.distributed as dist

from twinlakes.utils.checkpoint import save_checkpoint
from twinlakes.utils.common import StepTimer
from twinlakes.utils.train_utils import log_per_step, get_torch_dtype


def synchronized_batches(loader, device):
    """Stop all DDP ranks when any rank exhausts its input.

    This keeps optimizer/scheduler steps equal with streaming, uneven shards.
    It needs no extra process group or DDP join context.
    """
    iterator = iter(loader)
    distributed = dist.is_available() and dist.is_initialized()
    while True:
        try:
            batch = next(iterator)
            present = 1
        except StopIteration:
            batch, present = None, 0
        if distributed:
            flag = torch.tensor(present, device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            present = int(flag.item())
        if not present:
            return
        yield batch


class Executor:
    def __init__(self, global_step=0, device='cpu', runname=None):
        self.step = global_step
        self.device = torch.device(device)
        self.runname = runname
        self.scaler = None
        self.timer = StepTimer(global_step)

    def _forward(self, model, batch, dtype):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=dtype if value.is_floating_point() else value.dtype)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        kwargs = {name: batch[name] for name in ['input_ids', 'prompt_wavs_lengths', 'wavs_lengths', 'wavs', 'prompt_wavs']}
        kwargs['labels'] = batch['label_ids'].clone()
        kwargs.update(keys=batch['keys'], audio_pos=batch['audio_pos'])
        kwargs = move(kwargs)
        with torch.autocast(self.device.type, dtype=dtype, enabled=dtype != torch.float32):
            result = model(**kwargs)
        if not torch.isfinite(result['loss']).all():
            raise FloatingPointError('Nonfinite model loss')
        return result

    def train(self, model, optimizer, scheduler, train_data_loader, writer, configs, rank=0):
        model.train()
        dtype = get_torch_dtype(configs.get('dtype', 'fp32'))
        if self.scaler is None:
            self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        accum_grad = int(configs.get('accum_grad', 1))
        if accum_grad < 1:
            raise ValueError('accum_grad must be positive')
        clip = configs.get('grad_clip', 1.)
        max_steps = configs.get('max_steps', 0)
        optimizer.zero_grad(set_to_none=True)
        pending, losses = 0, {}

        def update():
            nonlocal pending, losses
            self.scaler.unscale_(optimizer)
            # loss/accum_grad was used for every microbatch; correct a short tail.
            if pending < accum_grad:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(accum_grad / pending)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(norm):
                self.scaler.step(optimizer)
                scheduler.step()
                self.step += 1
                info = {'tag': 'TRAIN', 'step': self.step, 'epoch': configs.get('epoch', 0),
                        'log_interval': configs.get('log_interval', 10),
                        'loss_dict': {key: value / pending for key, value in losses.items()},
                        'lrs': [group['lr'] for group in optimizer.param_groups], 'grad_norm': float(norm)}
                log_per_step(writer, info, self.timer)
                interval = configs.get('save_interval', 5000)
                if rank == 0 and interval and self.step % interval == 0 and configs.get('model_dir'):
                    path = Path(configs['model_dir']) / f"{configs.get('epoch', 0)}_{self.step}.pt"
                    save_checkpoint(model, path, {'epoch': configs.get('epoch', 0), 'step': self.step, 'epoch_complete': False},
                                    optimizer=optimizer, scheduler=scheduler, scaler=self.scaler)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending, losses = 0, {}

        # Synchronizing each backward also handles a partial accumulation window
        # without an unsynchronized final microbatch.
        for batch in synchronized_batches(train_data_loader, self.device):
            if max_steps and self.step >= max_steps:
                break
            result = self._forward(model, batch, dtype)
            self.scaler.scale(result['loss'] / accum_grad).backward()
            pending += 1
            for key, value in result.items():
                if value is not None:
                    losses[key] = losses.get(key, 0.) + float(value.detach())
            if pending == accum_grad:
                update()
        if pending:
            update()
        return self.step

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, configs):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        model.eval()
        dtype = get_torch_dtype(configs.get('dtype', 'fp32'))
        losses, samples = {}, 0
        for batch in cv_data_loader:
            result = self._forward(model, batch, dtype)
            size = len(batch['keys'])
            samples += size
            for key, value in result.items():
                if value is not None:
                    losses[key] = losses.get(key, 0.) + float(value) * size
        if not samples:
            raise ValueError('Validation loader produced no samples')
        losses = {key: value / samples for key, value in losses.items()}
        log_per_step(writer, {'tag': 'CV', 'step': self.step, 'epoch': configs.get('epoch', 0), 'loss_dict': losses})
        return losses['loss']
