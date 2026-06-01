import argparse
import importlib
import itertools
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile model latency and profiler FLOPs on the test set.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir).split('/')
    else:
        module_dir = os.path.dirname(config_path).split('/')
    module_path = module_dir[0]
    for part in module_dir[1:]:
        module_path = module_path + '.' + part
    importlib.import_module(module_path)


def prepare_test_cfg(cfg):
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    else:
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    return samples_per_gpu


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def init_single_process_group():
    if dist.is_available() and not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        port = os.environ.get('PROFILE_DIST_PORT',
                              str(int(os.environ.get('PORT', '28521')) + 1))
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://127.0.0.1:{port}',
            rank=0,
            world_size=1)


def run_once(model, data):
    with torch.inference_mode():
        return model(return_loss=False, rescale=True, **data)


def profile_flops(model, data):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False) as prof:
        run_once(model, data)
    return int(sum(evt.flops for evt in prof.key_averages() if evt.flops))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    init_single_process_group()

    samples_per_gpu = prepare_test_cfg(cfg)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    params = sum(p.numel() for p in model.module.parameters())
    latencies = []
    profiler_flops = None
    max_iters = args.warmup + args.samples

    for idx, data in enumerate(itertools.cycle(data_loader)):
        if idx >= max_iters:
            break
        cuda_synchronize()
        start = time.perf_counter()
        run_once(model, data)
        cuda_synchronize()
        elapsed = time.perf_counter() - start
        if idx >= args.warmup:
            latencies.append(elapsed)
            if profiler_flops is None:
                profiler_flops = profile_flops(model, data)

    if not latencies:
        raise RuntimeError('No samples were profiled. Check dataset length.')

    mean_latency = sum(latencies) / len(latencies)
    result = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        dataset_size=len(dataset),
        requested_samples=args.samples,
        requested_warmup=args.warmup,
        samples=len(latencies),
        warmup=args.warmup,
        batch_size=samples_per_gpu,
        params=params,
        profiler_flops=profiler_flops,
        single_batch_latency_s=latencies[0],
        mean_latency_s=mean_latency,
        median_latency_s=sorted(latencies)[len(latencies) // 2],
        fps=1.0 / mean_latency)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n')


if __name__ == '__main__':
    main()
