"""Train the current TwinLakes model with native PyTorch, on one device or DDP."""
import argparse
import copy
import datetime
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from twinlakes.utils.checkpoint import load_checkpoint, save_checkpoint
from twinlakes.utils.executor import Executor
from twinlakes.utils.scheduler import WarmupLR
from twinlakes.utils.train_utils import check_modify_and_save_config, get_torch_dtype


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--data_type', default='raw', choices=['raw', 'shard'])
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--cv_data')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', choices=['nccl', 'gloo'])
    parser.add_argument('--timeout', type=int, default=1800, help='DDP initialization/collective timeout in seconds')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--tensorboard_dir', help='Optional local TensorBoard log directory')
    parser.add_argument('--checkpoint', help='Resume weights and available optimizer/scheduler state')
    return parser.parse_args(argv)


def make_loader(dataset, args, seed):
    options = {'batch_size': None, 'num_workers': args.num_workers, 'pin_memory': args.pin_memory,
               'generator': torch.Generator().manual_seed(seed)}
    if args.num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=args.prefetch)
    return DataLoader(dataset, **options)


def latest_checkpoint(directory):
    candidates = []
    for path in Path(directory).glob('*.pt'):
        sidecar = path.with_suffix('.yaml')
        infos = yaml.safe_load(sidecar.read_text()) if sidecar.is_file() else {}
        try:
            step = int((infos or {}).get('step', path.stem.split('_')[-1]))
        except (TypeError, ValueError):
            continue
        candidates.append((step, path.stat().st_mtime_ns, path))
    return max(candidates)[-1] if candidates else None


def validate_model_config(config):
    required = {'temporal_model', 'vibevoice_path', 'dataset_conf', 'optim_conf', 'scheduler_conf'}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError('Current LMModel configuration is missing: ' + ', '.join(missing))


def model_parameter_summary(model):
    """Count unique Parameter objects per module, including tied LLM weights."""
    def count(parameters):
        parameters = list(parameters)
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        dtypes = ','.join(sorted({str(parameter.dtype).removeprefix('torch.') for parameter in parameters}))
        return dict(total=total, trainable=trainable, frozen=total-trainable, dtype=dtypes or '-')

    rows = {}
    direct = list(model.parameters(recurse=False))
    if direct:
        rows['<root>'] = count(direct)
    for name, module in model.named_children():
        rows[name] = count(module.parameters())
        if name == 'acoustic_tokenizer':
            for part in ('encoder', 'decoder'):
                submodule = getattr(module, part, None)
                if submodule is not None:
                    rows[f'{name}.{part}'] = count(submodule.parameters())
    rows['TOTAL'] = count(model.parameters())
    return rows


def log_model_info(model, configs, config_path, rank):
    if rank != 0:
        return
    logging.info('Training config: %s', Path(config_path).resolve())
    logging.info('LLM initialization path: %s', configs['temporal_model'])
    logging.info('Codec initialization path: %s', configs['vibevoice_path'])
    for name in ('lm', 'diffusion_head'):
        module = getattr(model, name, None)
        if module is None:
            continue
        config = getattr(module, 'config', None)
        dimensions = {key: getattr(config, key) for key in
                      ('hidden_size', 'num_hidden_layers', 'vocab_size', 'head_layers', 'head_ffn_ratio', 'latent_size')
                      if hasattr(config, key)}
        logging.info('Loaded %s: %s, dimensions=%s', name, type(module).__name__, dimensions)
    logging.info('Actual model structure:\n%s', model)
    lines = [f'{"Module":<36} {"Total":>17} {"Trainable":>17} {"Frozen":>17}  Dtype']
    for name, values in model_parameter_summary(model).items():
        lines.append(f'{name:<36} {values["total"]:>17,} {values["trainable"]:>17,} '
                     f'{values["frozen"]:>17,}  {values["dtype"]}')
    logging.info('Parameter summary (nested rows are breakdowns; TOTAL deduplicates shared parameters):\n%s',
                 '\n'.join(lines))


def main(argv=None):
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.num_workers < 0 or args.prefetch < 1:
        raise ValueError('num_workers must be nonnegative and prefetch must be positive')
    configs = yaml.safe_load(Path(args.config).read_text())
    validate_model_config(configs)
    configs = check_modify_and_save_config(args, configs)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', local_rank if world > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world > 1:
        backend = args.dist_backend or ('nccl' if device.type == 'cuda' else 'gloo')
        if backend == 'nccl' and device.type != 'cuda':
            raise ValueError('nccl requires CUDA; use gloo on CPU')
        dist.init_process_group(backend, timeout=datetime.timedelta(seconds=args.timeout))
    writer = None
    try:
        seed = int(configs.get('seed', 778)) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        from twinlakes.models.rq_transformer import LMModel
        from twinlakes.dataset.dataset import Dataset
        model = LMModel.from_audio_text_pretrained(configs)
        model.acoustic_tokenizer.requires_grad_(False)
        if configs.get('freeze_diffusion_head', False):
            model.diffusion_head.requires_grad_(False)
        dtype = get_torch_dtype(configs['dtype'])
        model = model.to(device=device, dtype=dtype)
        logging.info('Model parameters and floating inputs use %s on %s', dtype, device)
        log_model_info(model, configs, args.config, rank)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), **configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
        executor = Executor(device=device, runname=configs.get('run_name'))
        # Native GradScaler cannot unscale FP16 parameter gradients. Parameters
        # now use the configured dtype directly, without FP32 master weights.
        executor.scaler = torch.amp.GradScaler(device.type, enabled=False)
        checkpoint = args.checkpoint or latest_checkpoint(args.model_dir)
        infos = {}
        if checkpoint:
            infos = load_checkpoint(model, checkpoint, optimizer=optimizer, scheduler=scheduler, scaler=executor.scaler)
            executor.step = int(infos.get('step', 0))
        elif configs.get('init_checkpoint'):
            load_checkpoint(model, configs['init_checkpoint'])
        codec_path = configs['dataset_conf'].get('latent_codec_metadata')
        if args.data_type == 'shard' and configs['dataset_conf'].get('shard_latent') and codec_path:
            from twinlakes.dataset.latent_shard import load_codec_metadata
            codec = load_codec_metadata(codec_path)
            if codec['dimension'] != model.config.acoustic_vae_dim:
                raise ValueError('Cached latent dimension differs from model')
            for name in ('speech_scaling_factor', 'speech_bias_factor'):
                if not np.isclose(float(getattr(model, name)), codec[name], rtol=1e-5, atol=1e-6):
                    raise ValueError(f'Cached latent {name} differs from model')
        if world > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index] if device.type == 'cuda' else None,
                find_unused_parameters=True, broadcast_buffers=False)
        train_conf = copy.deepcopy(configs['dataset_conf'])
        train_conf['is_inference'] = False
        train_loader = make_loader(Dataset(args.data_type, args.train_data, model.module.tokenizer if world > 1 else model.tokenizer,
                                           train_conf, partition=True), args, seed)
        cv_loader = None
        if rank == 0:
            Path(args.model_dir).mkdir(parents=True, exist_ok=True)
            Path(args.model_dir, 'train.yaml').write_text(yaml.safe_dump(configs, allow_unicode=True))
            if args.tensorboard_dir:
                from tensorboardX import SummaryWriter
                writer = SummaryWriter(str(Path(args.tensorboard_dir) / Path(args.model_dir).name))
            if args.cv_data:
                cv_conf = copy.deepcopy(train_conf)
                cv_conf.update(shuffle=False, sort=False, list_shuffle=False, cycle=1)
                tokenizer = model.module.tokenizer if world > 1 else model.tokenizer
                cv_loader = make_loader(Dataset(args.data_type, args.cv_data, tokenizer, cv_conf, partition=False), args, seed)
        start_epoch = int(infos.get('epoch', -1)) + 1 if infos.get('epoch_complete', True) else int(infos['epoch'])
        for epoch in range(max(0, start_epoch), configs.get('max_epoch', 100)):
            if configs.get('max_steps') and executor.step >= configs['max_steps']:
                break
            configs['epoch'] = epoch
            executor.train(model, optimizer, scheduler, train_loader, writer, configs, rank)
            if rank == 0:
                loss = executor.cv(model, cv_loader, writer, configs) if cv_loader is not None else None
                save_checkpoint(model, Path(args.model_dir) / f'{epoch}_{executor.step}.pt',
                                {'epoch': epoch, 'step': executor.step, 'epoch_complete': True, 'cv_loss': loss},
                                optimizer=optimizer, scheduler=scheduler, scaler=executor.scaler)
            if world > 1:
                dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
