"""Per-component FLOPs and latency profiler for Doracamom.

Unlike ``profile_latency_flops.py`` (whole-model only, and FLOPs via
``torch.profiler(with_flops=True)`` which silently skips the custom 3D
deformable-attention CUDA op), this script attributes both *latency* and
*FLOPs* to each architectural component, so the compute bottleneck is visible.

Design (see docs plan): instrumentation is done entirely with runtime hooks
registered from here -- there are **zero edits to any model/forward file**.

  * Latency  : CUDA-event ``forward_pre_hook``/``forward_hook`` per component
               module (single ``cuda.synchronize`` per iter -> accurate GPU
               time). On CPU it falls back to ``perf_counter`` wall time.
  * FLOPs    : trace-free hook-based MAC counter on every leaf module
               (Conv/Linear/ConvTranspose handlers), plus an analytical handler
               for the custom ``MSDeformableAttention3DV1`` sampling/aggregation
               kernel (its 3 Linear projections are already counted as leaves).
               FLOPs are reported as ``2 * MACs``.

The "Coarse Voxel Queries Generator" is a code block in ``head.forward`` (not a
single module); its latency is recovered by residual
(``head - transformer - decoder``) and its FLOPs from ``reduc_conv``.

Results are written as a detailed markdown report (+ JSON) under
``profiles/<config-stem>_<timestamp>/component_profile.{md,json}``.
"""

import argparse
import itertools
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

# Reuse helpers from the whole-model profiler (no edits to it). The script's own
# directory is on sys.path[0], so this import resolves when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_latency_flops import (  # noqa: E402
    cuda_synchronize, import_plugins, init_single_process_group,
    prepare_test_cfg, run_once)

# Detected by class name (not import) so this script loads even where the custom
# deform-attn .so cannot be imported (e.g. a CPU login node).
DEFORM_ATTN_CLS = 'MSDeformableAttention3DV1'


# --- component -> module mapping ------------------------------------------
# Paths are relative to the detector (``model.module``). Verified against the
# inference path: simple_test -> extract_feat -> pts_bbox_head.forward ->
# transformer.forward.
RESIDUAL = '__RESIDUAL_HEAD__'

LATENCY_GROUPS = OrderedDict([
    ('Camera Encoder', ['img_backbone', 'img_neck']),
    ('4D Radar Encoder', ['pts_voxel_layer', 'pts_voxel_encoder',
                          'pts_middle_encoder', 'pts_backbone', 'pts_neck']),
    ('Coarse Voxel Queries Generator', RESIDUAL),
    ('Voxel Queries Encoder', ['pts_bbox_head.transformer.cam_encoder']),
    ('Dual-branch Temporal Encoder',
     ['pts_bbox_head.transformer.temporal_encoder']),
    ('Cross-Modal BEV-Voxel Fusion',
     ['pts_bbox_head.transformer.voxel_decoder',
      'pts_bbox_head.transformer.occbevfusion2d']),
    ('Detection Head', ['pts_bbox_head.decoder']),
    ('BEV-seg Aux', ['pts_bbox_head.transformer.bev_seg_decoder']),
])

# Modules to time. Includes the two "whole" modules needed for the residual.
LATENCY_TARGETS = sorted({
    'img_backbone', 'img_neck',
    'pts_voxel_layer', 'pts_voxel_encoder', 'pts_middle_encoder',
    'pts_backbone', 'pts_neck',
    'pts_bbox_head', 'pts_bbox_head.transformer',
    'pts_bbox_head.transformer.cam_encoder',
    'pts_bbox_head.transformer.temporal_encoder',
    'pts_bbox_head.transformer.voxel_decoder',
    'pts_bbox_head.transformer.occbevfusion2d',
    'pts_bbox_head.transformer.bev_seg_decoder',
    'pts_bbox_head.decoder', 'pts_bbox_head.reduc_conv',
})

# leaf-FLOPs / params aggregation: component -> longest-prefix candidates.
PREFIX_MAP = OrderedDict([
    ('Camera Encoder', ['img_backbone', 'img_neck']),
    ('4D Radar Encoder', ['pts_voxel_encoder', 'pts_middle_encoder',
                          'pts_backbone', 'pts_neck', 'pts_voxel_layer']),
    ('Coarse Voxel Queries Generator', ['pts_bbox_head.reduc_conv']),
    ('Voxel Queries Encoder', ['pts_bbox_head.transformer.cam_encoder']),
    ('Dual-branch Temporal Encoder',
     ['pts_bbox_head.transformer.temporal_encoder']),
    ('Cross-Modal BEV-Voxel Fusion',
     ['pts_bbox_head.transformer.voxel_decoder',
      'pts_bbox_head.transformer.occbevfusion2d']),
    ('Detection Head', ['pts_bbox_head.decoder']),
    ('BEV-seg Aux', ['pts_bbox_head.transformer.bev_seg_decoder']),
])
OTHER = 'Other / Residual'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-component FLOPs and latency profiler.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument(
        '--output-dir', default=None,
        help='Defaults to profiles/<config-stem>_<timestamp>/')
    parser.add_argument(
        '--deform-macs-per-elem', type=float, default=5.0,
        help='MACs per (token, head, level, point, head_dim) for the custom 3D '
             'deform-attn sampling kernel: ~4 bilinear taps + 1 weighting. '
             'Reported FLOPs use 2x this. See header.')
    return parser.parse_args()


# --- module resolution ----------------------------------------------------
def resolve(model, dotted):
    """Resolve a dotted attribute path on the detector; None if absent."""
    obj = model
    for part in dotted.split('.'):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if isinstance(obj, nn.Module) else None


def longest_prefix_component(name, prefix_map):
    best, best_len = OTHER, -1
    for comp, prefixes in prefix_map.items():
        for pre in prefixes:
            if (name == pre or name.startswith(pre + '.')) and len(pre) > best_len:
                best, best_len = comp, len(pre)
    return best


# --- FLOPs (MAC) handlers --------------------------------------------------
def _to_tensor(x):
    if isinstance(x, (list, tuple)):
        return _to_tensor(x[0]) if x else None
    return x if torch.is_tensor(x) else None


def leaf_macs(module, inp, out):
    """MACs for a leaf module from its I/O shapes (bias adds ignored)."""
    out_t = _to_tensor(out)
    in_t = _to_tensor(inp)
    if out_t is None:
        return 0.0
    if isinstance(module, nn.Linear):
        return float(out_t.numel()) * module.in_features
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        k = 1
        for s in module.kernel_size:
            k *= s
        return float(out_t.numel()) * (module.in_channels // module.groups) * k
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d,
                           nn.ConvTranspose3d)):
        if in_t is None:
            return 0.0
        k = 1
        for s in module.kernel_size:
            k *= s
        # MACs scale with the *input* volume for transposed conv.
        return float(in_t.numel()) * (module.out_channels // module.groups) * k
    return 0.0


class FlopMeter:
    """Hook-based per-module MAC counter (reset each iteration)."""

    def __init__(self, model, deform_macs_per_elem):
        self.per_module = defaultdict(float)
        self.handles = []
        self.deform_c = float(deform_macs_per_elem)
        for name, module in model.named_modules():
            if type(module).__name__ == DEFORM_ATTN_CLS:
                self.handles.append(
                    module.register_forward_hook(self._deform(name, module)))
                continue  # not a leaf; its Linear children are counted below
            if len(list(module.children())) == 0:
                self.handles.append(
                    module.register_forward_hook(self._leaf(name, module)))

    def _leaf(self, name, module):
        def hook(m, inp, out):
            self.per_module[name] += leaf_macs(m, inp, out)
        return hook

    def _deform(self, name, module):
        def hook(m, inp, out):
            out_t = _to_tensor(out)
            if out_t is None:
                return
            tokens = out_t.numel() // m.embed_dims
            head_dim = m.embed_dims // m.num_heads
            macs = (tokens * m.num_heads * m.num_levels * m.num_points
                    * head_dim * self.deform_c)
            self.per_module[name] += float(macs)
        return hook

    def snapshot_and_reset(self):
        snap = dict(self.per_module)
        self.per_module = defaultdict(float)
        return snap

    def remove(self):
        for h in self.handles:
            h.remove()


# --- latency meter ---------------------------------------------------------
class LatencyMeter:
    """CUDA-event (or perf_counter on CPU) timing per named module."""

    def __init__(self, named_targets, use_cuda):
        self.use_cuda = use_cuda
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self._cur = {}
        self.handles = []
        for name, module in named_targets.items():
            self.handles.append(
                module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(m, inp):
            if self.use_cuda:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._cur[name] = [ev, None]
            else:
                self._cur[name] = [time.perf_counter(), None]
        return hook

    def _post(self, name):
        def hook(m, inp, out):
            if name not in self._cur:
                return
            if self.use_cuda:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._cur[name][1] = ev
            else:
                self._cur[name][1] = time.perf_counter()
        return hook

    def collect(self):
        """Call after cuda.synchronize(); accumulates ms for this iteration."""
        for name, (start, end) in self._cur.items():
            if end is None:
                continue
            ms = (start.elapsed_time(end) if self.use_cuda
                  else (end - start) * 1e3)
            self.totals[name] += ms
            self.counts[name] += 1
        self._cur = {}

    def mean_ms(self, name):
        c = self.counts.get(name, 0)
        return self.totals[name] / c if c else 0.0

    def remove(self):
        for h in self.handles:
            h.remove()


# --- aggregation -----------------------------------------------------------
def component_latencies(meter):
    head = meter.mean_ms('pts_bbox_head')
    trans = meter.mean_ms('pts_bbox_head.transformer')
    dec = meter.mean_ms('pts_bbox_head.decoder')
    out = OrderedDict()
    for comp, paths in LATENCY_GROUPS.items():
        if paths == RESIDUAL:
            out[comp] = max(head - trans - dec, 0.0)
        else:
            out[comp] = sum(meter.mean_ms(p) for p in paths)
    return out


def component_macs(mean_macs_per_module):
    out = OrderedDict((c, 0.0) for c in PREFIX_MAP)
    out[OTHER] = 0.0
    for name, macs in mean_macs_per_module.items():
        out[longest_prefix_component(name, PREFIX_MAP)] += macs
    return out


def component_params(model):
    out = OrderedDict((c, 0) for c in PREFIX_MAP)
    out[OTHER] = 0
    for name, p in model.named_parameters():
        out[longest_prefix_component(name, PREFIX_MAP)] += p.numel()
    return out


# --- report ----------------------------------------------------------------
def fmt(v, nd=2):
    if v is None:
        return 'n/a'
    if abs(v) >= 1000:
        return f'{v:,.{nd}f}'
    return f'{v:.{nd}f}'


def build_report(meta, overall, rows, recon):
    lines = [
        '# Doracamom Per-Component Profiling Report', '',
        f"- Config: `{meta['config']}`",
        f"- Checkpoint: `{meta['checkpoint']}`",
        f"- Device: `{meta['device']}`",
        f"- Dataset size: `{meta['dataset_size']}`",
        f"- Batch size: `{meta['batch_size']}`",
        f"- Warmup / profiled samples: `{meta['warmup']}` / `{meta['samples']}`",
        f"- Generated: `{meta['timestamp']}`",
        '',
        '## Overall Metrics', '',
        '| Metric | Value |', '| :--- | ---: |',
        f"| Total params (M) | {fmt(overall['params_m'])} |",
        f"| End-to-end mean latency (ms) | {fmt(overall['mean_latency_ms'])} |",
        f"| End-to-end median latency (ms) | {fmt(overall['median_latency_ms'])} |",
        f"| FPS | {fmt(overall['fps'])} |",
        f"| Total FLOPs (GFLOPs) | {fmt(overall['total_gflops'])} |",
        f"| Total MACs (GMACs) | {fmt(overall['total_gmacs'])} |",
        '',
        '## Per-Component Breakdown', '',
        '_Sorted by latency (the time bottleneck). FLOPs = 2 x MACs._', '',
        ('| Component | Latency (ms) | Lat % | FLOPs (GFLOPs) | FLOPs % '
         '| Params (M) |'),
        '| :--- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {fmt(r['latency_ms'])} | {fmt(r['latency_pct'])} "
            f"| {fmt(r['gflops'])} | {fmt(r['flops_pct'])} "
            f"| {fmt(r['params_m'])} |")
    lines += [
        '',
        '## Reconciliation', '',
        '| Check | Value |', '| :--- | ---: |',
        f"| Sum of component latency (ms) | {fmt(recon['sum_latency_ms'])} |",
        f"| End-to-end latency, hooked (ms) | {fmt(recon['hooked_latency_ms'])} |",
        f"| Unattributed latency (ms) | {fmt(recon['unattributed_ms'])} |",
        f"| Sum of component FLOPs (GFLOPs) | {fmt(recon['sum_gflops'])} |",
        '',
        '## Methodology / Caveats', '',
        '- Latency is GPU time via CUDA events (one `synchronize` per iteration). '
        'CPU-only work (e.g. voxelization) is captured only for its GPU portion.',
        '- FLOPs come from a trace-free hook counter (Conv/Linear/ConvTranspose) '
        'reported as `2 x MACs`; BatchNorm/activation/interp are treated as ~0.',
        '- The custom 3D deformable-attention sampling/aggregation is counted '
        f"analytically at {meta['deform_c']} MACs per "
        '(token x head x level x point x head_dim); its Linear projections are '
        'counted exactly as leaves.',
        '- "Coarse Voxel Queries Generator" latency = head - transformer - '
        'decoder (residual); its FLOPs come from `reduc_conv`.',
        '- Profiling uses the inference path (`return_loss=False`).',
        '',
    ]
    return '\n'.join(lines)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    init_single_process_group()

    samples_per_gpu = prepare_test_cfg(cfg)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = MMDataParallel(model.cuda(), device_ids=[0])
        detector = model.module
    else:
        detector = model
    model.eval()

    params_total = sum(p.numel() for p in detector.parameters())

    # Pass 1: clean end-to-end latency (no hooks) -- the headline numbers.
    clean_latencies = []
    max_iters = args.warmup + args.samples
    for idx, data in enumerate(itertools.cycle(data_loader)):
        if idx >= max_iters:
            break
        cuda_synchronize()
        t0 = time.perf_counter()
        run_once(model, data)
        cuda_synchronize()
        if idx >= args.warmup:
            clean_latencies.append(time.perf_counter() - t0)
    if not clean_latencies:
        raise RuntimeError('No samples profiled. Check dataset length.')

    # Pass 2: hooked pass for per-component latency + FLOPs.
    flop_meter = FlopMeter(detector, args.deform_macs_per_elem)
    lat_targets = {name: m for name in LATENCY_TARGETS
                   if (m := resolve(detector, name)) is not None}
    lat_meter = LatencyMeter(lat_targets, use_cuda)

    macs_iters = []
    hooked_latencies = []
    for idx, data in enumerate(itertools.cycle(data_loader)):
        if idx >= max_iters:
            break
        flop_meter.snapshot_and_reset()
        cuda_synchronize()
        t0 = time.perf_counter()
        run_once(model, data)
        cuda_synchronize()
        dt = time.perf_counter() - t0
        snap = flop_meter.snapshot_and_reset()
        if idx >= args.warmup:
            lat_meter.collect()
            macs_iters.append(snap)
            hooked_latencies.append(dt)
        else:
            lat_meter._cur = {}
    flop_meter.remove()
    lat_meter.remove()

    # mean MACs per module across profiled iters
    mean_macs = defaultdict(float)
    for snap in macs_iters:
        for k, v in snap.items():
            mean_macs[k] += v
    n = len(macs_iters)
    for k in mean_macs:
        mean_macs[k] /= n

    comp_lat = component_latencies(lat_meter)
    comp_macs = component_macs(mean_macs)
    comp_params = component_params(detector)

    total_macs = sum(mean_macs.values())
    total_flops = 2.0 * total_macs
    sum_lat = sum(comp_lat.values())
    clean_mean = sum(clean_latencies) / len(clean_latencies) * 1e3
    clean_median = sorted(clean_latencies)[len(clean_latencies) // 2] * 1e3
    hooked_mean = sum(hooked_latencies) / len(hooked_latencies) * 1e3

    # assemble rows (every component appears; latency from comp_lat, flops from
    # comp_macs, params from comp_params) + the OTHER row for unmatched flops.
    row_names = list(LATENCY_GROUPS.keys()) + [OTHER]
    rows = []
    for name in row_names:
        lat = comp_lat.get(name, 0.0)
        macs = comp_macs.get(name, 0.0)
        gflops = 2.0 * macs / 1e9
        rows.append(dict(
            name=name,
            latency_ms=lat,
            latency_pct=100.0 * lat / sum_lat if sum_lat else 0.0,
            gflops=gflops,
            flops_pct=100.0 * (2.0 * macs) / total_flops if total_flops else 0.0,
            params_m=comp_params.get(name, 0) / 1e6))
    rows.sort(key=lambda r: r['latency_ms'], reverse=True)

    meta = dict(
        config=args.config, checkpoint=args.checkpoint,
        device=(torch.cuda.get_device_name(0) if use_cuda else 'cpu'),
        dataset_size=len(dataset), batch_size=samples_per_gpu,
        warmup=args.warmup, samples=len(clean_latencies),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        deform_c=args.deform_macs_per_elem)
    overall = dict(
        params_m=params_total / 1e6, mean_latency_ms=clean_mean,
        median_latency_ms=clean_median, fps=1000.0 / clean_mean,
        total_gflops=total_flops / 1e9, total_gmacs=total_macs / 1e9)
    recon = dict(
        sum_latency_ms=sum_lat, hooked_latency_ms=hooked_mean,
        unattributed_ms=hooked_mean - sum_lat,
        sum_gflops=sum(r['gflops'] for r in rows))

    stem = Path(args.config).stem
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        'profiles') / f'{stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = build_report(meta, overall, rows, recon)
    (out_dir / 'component_profile.md').write_text(report_md)
    (out_dir / 'component_profile.json').write_text(json.dumps(dict(
        meta=meta, overall=overall, components=rows, reconciliation=recon),
        indent=2, sort_keys=True) + '\n')

    print(report_md)
    print(f'\nWrote {out_dir}/component_profile.md')
    print(f'Wrote {out_dir}/component_profile.json')


if __name__ == '__main__':
    main()
