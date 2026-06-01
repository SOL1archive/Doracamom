# Code Review: `d3238c` vs current HEAD

- **Date:** 2026-06-01
- **Base commit:** `d3238c7` ("update")
- **HEAD commit:** `5090ae6` ("Add Slurm metrics workflow")
- **Range:** 8 commits (`d3238c..HEAD`)

## Question under review

> Compare `d3238c` and the current HEAD, and check whether
> 1. the core implementation logic has changed,
> 2. K-Radar, TJ4D, VoD integrations are implemented the right way,
> 3. FLOPs and latency profilings are done reasonably.

## What changed between the two commits

```
.gitignore                                          |   6 +-
README.md                                           |  29 +-
deform_attn_3d/setup.py                             |   2 +-
job-scripts/*.sh (new launch scripts)               | many
mmdetection3d/mmdet3d/core/evaluation/__init__.py   |   4 +-
mmdetection3d/.../kitti_utils/eval_tj4d.py          |   2 +-
mmdetection3d/mmdet3d/datasets/kitti_dataset_tj4d.py|  22 +-
projects/configs/Doracamom/Doracamom_KRadar.py      | 134 (new)
projects/mmdet3d_plugin/rc_detsoc/modules/
    multi_scale_deformable_attn_3D_custom_function.py|  26 +-
tools/analysis_tools/profile_latency_flops.py       | 175 (new)
tools/analysis_tools/make_basic_metrics_report.py   | 108 (new)
tools/create_data.py                                |  56 +-
tools/data_converter/kradar_converter.py            | 321 (new)
tools/{dist_test,dist_train}.sh, test.py, train.py  | path/guard fixes
```

No model / head / loss / transformer compute file was modified. The diff is: K-Radar
support (converter + config + launch scripts), TJ4D evaluation fixes, path/portability
hardening, and a FLOPs/latency profiling + reporting workflow.

---

## 1. Has the core implementation logic changed? — No

The detector, dense heads, losses, transformer, and the 3D deformable-attention CUDA math
are **untouched**. The only edits near the core are mechanical portability fixes:

- **`multi_scale_deformable_attn_3D_custom_function.py`** — the hardcoded
  `sys.path.append("/mnt/zhenglianqing/Doracamom/deform_attn_3d/")` was replaced with a
  repo-relative `Path(__file__).resolve().parents[4] / "deform_attn_3d"`. The imported
  `ext_module` and every kernel are identical.
- **`deform_attn_3d/setup.py`** — added `FORCE_CUDA` so the op can be built on a CPU-only
  login node.
- **`tools/train.py` / `tools/test.py`** — removed hardcoded `sys.path.insert(...)`, added
  an `MPLCONFIGDIR` default, and (importantly) changed `if 'load_img_from' in cfg:` →
  `if cfg.get('load_img_from', None) is not None:` (and the three sibling `load_*` keys).
  This is a **correct, meaningful fix**: a config key that is present but `None`
  (e.g. K-Radar's `load_pts_from = None`) would previously have triggered
  `torch.load(None)`.
- **`tools/dist_train.sh` / `tools/dist_test.sh`** — add `deform_attn_3d` to `PYTHONPATH`
  and absolutize the script paths.

**Verdict:** core logic preserved; changes are environment/portability hardening. Good.

---

## 2. Are the K-Radar / TJ4D / VoD integrations done the right way? — Mixed

### VoD — unchanged
`git diff d3238c HEAD -- Doracamom_vod.py` is empty. VoD is the original reference path.
The README only adds sanity-run commands. Fine.

### TJ4D — genuine, correct bug fixes
- `kitti_dataset_tj4d.py:53` — `CLASSES` capitalized to `('Car','Pedestrian','Cyclist','Truck')`
  so the dataset class names match the capitalized `name` field in annotations and the
  config's `class_names`. (Order differs from the config, but that is harmless because
  `classes=class_names` is passed in the data config and overrides the class attribute.)
- `eval_tj4d.py:745` — guards `len(anno['alpha']) > 0` before indexing, avoiding a crash
  on empty-GT frames.
- `kitti_dataset_tj4d.py:284` — `outputs['bbox_results']` access is now guarded;
  `label_preds` shape fixed from `[0,4]` → `[0]` (lines 658/701); `eval_types` now respects
  the `metric` argument instead of always forcing `['bbox','bev','3d']`.

All reasonable. ✓

### K-Radar — wired plausibly, but NOT verified end-to-end
The approach (adapt K-Radar into KITTI-format infos and reuse the TJ4D dataset/eval) is
sound, and conversion **runs** (8 train / 2 val toy infos created). However:

**(a) Training does not work yet.** The only in-repo training attempt
(`logs/slurm/doracamom-kradar-1510799.err`) ends in an NCCL `BROADCAST` watchdog timeout
after 1800 s with **zero iterations logged** and **no Python traceback** — i.e. a pure
distributed collective hang, not a shape crash. The most common cause is a DDP
first-iteration desync: `find_unused_parameters` defaults to `False`
(`projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py:74`) and no Doracamom config sets
it. (TJ4D/VoD were only ever *inferenced* here from downloaded checkpoints, never trained,
so they are not counter-evidence.) See the K-Radar fix section below.

**(b) BEV-resolution inconsistency.** For K-Radar,
`pts_middle_encoder.output_shape=[200,450]` (0.16 m grid) → SECOND/SECONDFPN → `[100,225]`,
but the model is given `bev_h=160, bev_w=360` (`Doracamom_KRadar.py:52`) and the head uses
`bev_h_=80, bev_w_=180`. These are 0.32 m / 0.2 m / 0.4 m grids that do not line up the way
the TJ4D config does (where the model-level BEV equals the radar-neck output). The image-BEV
and radar-BEV that feed `OccbevFusion2D` will not share spatial dims. Because the run hung
before completing a forward, this was never exercised — it must be re-derived from
`point_cloud_range` / `voxel_size` and validated once DDP is fixed.

**(c) Unvalidated converter heuristics** (`tools/data_converter/kradar_converter.py:127-187`):
- Fixed 0.4 m cube spacing with `y -= 80`, `z -= 30` offsets and `np.flip(cube, axis=0)`
  on the z-axis (mirrors z unless the cube is genuinely stored descending).
- 99th-percentile power thresholding + `max_points` cap to turn the dense cube into points.
- A fabricated `P2` (focal = 700), identity rectification, and whole-image 2D boxes —
  acceptable only because this is a radar/BEV detector that does not use real 2D image AP.
- Radar power is written to **feature index 5** (`[x,y,z,0,0,power,0,0]`) and the config's
  `use_dim=[0,1,2,3,5]` then feeds it into TJ4D's **velocity/Doppler** slot, while the
  RCS slot (index 3) stays 0. This is likely a channel-semantics swap — power ≈ RCS
  belongs at index 3, with the velocity slot left 0 (the `radar_zyx_cube` has no Doppler).

**Verdict:** TJ4D fixes correct; VoD untouched; **K-Radar plausible but unverified** —
training currently hangs, and the converter/config carry several coordinate, channel, and
resolution assumptions that must be validated.

---

## 3. Are the FLOPs and latency profilings reasonable? — Latency yes; FLOPs & report caveated

`tools/analysis_tools/profile_latency_flops.py`:

- **Latency methodology is sound:** warmup loop, `torch.cuda.synchronize()` on both sides of
  `perf_counter`, `torch.inference_mode()`, `model.eval()`, loader cycling, mean/median/FPS.
  Measured TJ4D numbers (~0.29 s, 3.4 FPS, 54.1 M params on RTX 3090) are believable. ✓
- **FLOPs are an undercount.** `torch.profiler(with_flops=True)` only attributes FLOPs to a
  handful of ops (mm/addmm/bmm/conv). The **custom CUDA 3D deformable-attention op is not
  counted at all**, nor are norms/activations/elementwise ops. So `profiler_flops ≈ 1.33 TFLOPs`
  is a floor, not the true cost, and it is measured from a single sample. Fine as a rough
  figure if labeled as such; cross-check with `fvcore` / `mmcv get_model_complexity_info`.
- **The auto-report parses the wrong accuracy table.** `make_basic_metrics_report.py:19-33`
  grabs the *last* `{...}` dict in the stdout log, which is the KITTI **2D image-plane AP**
  table — all zeros for a radar 3D detector — instead of the 3D/BEV AP that matters. The
  generated `basic_metrics_report.md` therefore shows every class at `0`, which is
  misleading. It should target the `*_3D_*` / `*_BEV_*` keys.
- Profiling was run only on a **41-frame TJ4D sample subset** with the downloaded
  checkpoint; no K-Radar / VoD profiling exists yet.

**Verdict:** latency profiling is trustworthy; the FLOPs number is a partial undercount that
should be caveated/cross-checked; the report's accuracy section currently parses the wrong
(2D) metrics and reports all zeros.

---

## Recommended follow-ups (in priority order)

1. **Unblock K-Radar training** — set `find_unused_parameters = True` (standard DDP-hang
   remedy; safe), then re-run the toy train to confirm it gets past iteration 0. A
   single-GPU smoke run is the decisive isolation test (it removes DDP entirely).
2. **Validate the K-Radar BEV dimension chain** (`bev_h/bev_w` vs radar scatter/neck) and the
   radar feature-channel placement (power → RCS slot, index 3).
3. **Fix `load_accuracy`** in `make_basic_metrics_report.py` to select the 3D/BEV AP table.
4. **Caveat / cross-check FLOPs** — label `profiler_flops` as a lower bound and validate with
   an independent counter.
