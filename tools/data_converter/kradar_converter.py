# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
from pathlib import Path

import mmcv
import numpy as np

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


CLASS_MAP = {
    'Sedan': 'Car',
    'Van': 'Car',
    'Bus or Truck': 'Truck',
    'Bus': 'Truck',
    'Truck': 'Truck',
    'Pedestrian': 'Pedestrian',
    'Bicycle': 'Cyclist',
    'Cyclist': 'Cyclist',
    'Motorcycle': 'Cyclist',
}

VELO_TO_CAM = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32)


def create_kradar_infos(root_path,
                        out_dir,
                        info_prefix='kradar',
                        coord_type='radar',
                        z_offset=0.0,
                        cube_percentile=99.0,
                        max_points=20000,
                        val_ratio=0.2,
                        train_split=None,
                        val_split=None):
    root_path = Path(root_path)
    out_dir = Path(out_dir)
    (out_dir / 'training' / 'velodyne').mkdir(parents=True, exist_ok=True)
    (out_dir / 'training' / 'image_2').mkdir(parents=True, exist_ok=True)

    samples = _collect_samples(root_path)
    train_keys = _read_split(train_split)
    val_keys = _read_split(val_split)
    if not train_keys and not val_keys:
        split_at = int(len(samples) * (1 - val_ratio))
        train_samples, val_samples = samples[:split_at], samples[split_at:]
    else:
        train_samples = [s for s in samples if _sample_key(s) in train_keys]
        val_samples = [s for s in samples if _sample_key(s) in val_keys]

    train_infos = _convert_samples(train_samples, out_dir, 0, coord_type,
                                   z_offset, cube_percentile, max_points)
    val_infos = _convert_samples(val_samples, out_dir, len(train_infos),
                                 coord_type, z_offset, cube_percentile,
                                 max_points)

    mmcv.dump(train_infos, out_dir / f'{info_prefix}_infos_train.pkl')
    mmcv.dump(val_infos, out_dir / f'{info_prefix}_infos_val.pkl')
    mmcv.dump(train_infos + val_infos,
              out_dir / f'{info_prefix}_infos_trainval.pkl')
    print(f'K-Radar train infos: {len(train_infos)}')
    print(f'K-Radar val infos: {len(val_infos)}')


def _collect_samples(root_path):
    seq_dirs = sorted(p for p in root_path.rglob('*')
                      if p.is_dir() and (p / 'info_label').is_dir())
    samples = []
    for seq_dir in seq_dirs:
        for label_path in sorted((seq_dir / 'info_label').glob('*.txt')):
            frame_id = label_path.stem
            samples.append(
                dict(
                    seq=seq_dir.name,
                    seq_dir=seq_dir,
                    frame_id=frame_id,
                    label_path=label_path,
                    image_path=_find_frame_file(seq_dir, frame_id,
                                                ['cam-front', 'cam_front']),
                    cube_path=_find_frame_file(seq_dir, frame_id,
                                               ['radar_zyx_cube'])))
    return samples


def _convert_samples(samples, out_dir, start_idx, coord_type, z_offset,
                     cube_percentile, max_points):
    infos = []
    for offset, sample in enumerate(mmcv.track_iter_progress(samples)):
        sample_idx = start_idx + offset
        points = _load_cube_points(sample['cube_path'], cube_percentile,
                                   max_points)
        point_path = out_dir / 'training' / 'velodyne' / f'{sample_idx:06d}.bin'
        points.astype(np.float32).tofile(point_path)

        image_shape, image_rel_path = _link_image(sample['image_path'], out_dir,
                                                  sample_idx)
        calib = _make_calib(image_shape)
        boxes_lidar, names = _load_labels(sample['label_path'], sample,
                                          coord_type, z_offset)
        annos = _make_annos(boxes_lidar, names, image_shape)
        infos.append(
            dict(
                image=dict(
                    image_idx=sample_idx,
                    image_shape=np.array(image_shape, dtype=np.int32),
                    image_path=image_rel_path),
                point_cloud=dict(
                    num_features=8,
                    velodyne_path=str(
                        Path('training') / 'velodyne' /
                        f'{sample_idx:06d}.bin')),
                calib=calib,
                annos=annos))
    return infos


def _load_cube_points(cube_path, cube_percentile, max_points):
    if cube_path is None:
        raise FileNotFoundError('Missing radar_zyx_cube frame for K-Radar.')
    if cube_path.suffix == '.npy':
        cube = np.load(cube_path)
    else:
        if loadmat is None:
            raise ImportError('scipy is required to read K-Radar .mat cubes.')
        mat = loadmat(cube_path)
        cube = _pick_cube_array(mat, cube_path)
    cube = np.flip(cube, axis=0)
    cube = np.maximum(cube, 0)
    nonzero = cube[cube > 0]
    if nonzero.size == 0:
        return np.zeros((0, 8), dtype=np.float32)

    threshold = np.percentile(nonzero, cube_percentile)
    z_idx, y_idx, x_idx = np.where(cube >= threshold)
    power = cube[z_idx, y_idx, x_idx]
    if power.size > max_points:
        keep = np.argpartition(power, -max_points)[-max_points:]
        z_idx, y_idx, x_idx, power = z_idx[keep], y_idx[keep], x_idx[
            keep], power[keep]

    x = x_idx.astype(np.float32) * 0.4
    y = y_idx.astype(np.float32) * 0.4 - 80.0
    z = z_idx.astype(np.float32) * 0.4 - 30.0
    zeros = np.zeros_like(x, dtype=np.float32)
    return np.stack([x, y, z, zeros, zeros, power, zeros, zeros], axis=1)


def _load_labels(label_path, sample, coord_type, z_offset):
    calib_offset = _load_calib_offset(sample['seq_dir'], z_offset)
    boxes, names = [], []
    with open(label_path) as f:
        lines = f.readlines()[1:]
    for line in lines:
        values = [v.strip() for v in line.split(',')]
        if not values or values[0] != '*':
            continue
        offset = 1 if len(values) == 11 else 0
        raw_name = values[2 + offset]
        name = CLASS_MAP.get(raw_name)
        if name is None:
            continue
        x = float(values[3 + offset])
        y = float(values[4 + offset])
        z = float(values[5 + offset])
        yaw = np.deg2rad(float(values[6 + offset]))
        length = 2 * float(values[7 + offset])
        width = 2 * float(values[8 + offset])
        height = 2 * float(values[9 + offset])
        if coord_type == 'radar':
            x += calib_offset[0]
            y += calib_offset[1]
            z += calib_offset[2]
        boxes.append([x, y, z, length, width, height, yaw])
        names.append(name)
    return np.array(boxes, dtype=np.float32), np.array(names)


def _pick_cube_array(mat, cube_path):
    for key in ('arr_zyx', 'radar_zyx_cube', 'cube', 'arr'):
        if key in mat:
            return np.asarray(mat[key])

    candidates = [
        value for key, value in mat.items()
        if not key.startswith('__') and isinstance(value, np.ndarray)
        and value.ndim >= 3
    ]
    if candidates:
        return np.asarray(max(candidates, key=lambda arr: arr.size))
    raise KeyError(f'No 3D radar cube array found in {cube_path}.')


def _make_annos(boxes_lidar, names, image_shape):
    num = len(names)
    if num == 0:
        return dict(
            name=np.array([]),
            truncated=np.array([]),
            occluded=np.array([]),
            alpha=np.array([]),
            bbox=np.zeros((0, 4), dtype=np.float32),
            dimensions=np.zeros((0, 3), dtype=np.float32),
            location=np.zeros((0, 3), dtype=np.float32),
            rotation_y=np.array([]),
            difficulty=np.array([], dtype=np.int32),
            score=np.array([]))

    centers_cam = boxes_lidar[:, :3] @ VELO_TO_CAM[:3, :3].T
    dims_cam = boxes_lidar[:, [4, 5, 3]]
    h, w = image_shape[:2]
    return dict(
        name=names,
        truncated=np.zeros(num, dtype=np.float32),
        occluded=np.zeros(num, dtype=np.int64),
        alpha=np.zeros(num, dtype=np.float32),
        bbox=np.tile(np.array([[0, 0, w - 1, h - 1]], dtype=np.float32),
                     (num, 1)),
        dimensions=dims_cam.astype(np.float32),
        location=centers_cam.astype(np.float32),
        rotation_y=boxes_lidar[:, 6].astype(np.float32),
        difficulty=np.zeros(num, dtype=np.int32),
        score=np.ones(num, dtype=np.float32))


def _make_calib(image_shape):
    h, w = image_shape[:2]
    focal = 700.0
    p2 = np.array([[focal, 0, w / 2, 0], [0, focal, h / 2, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]],
                  dtype=np.float32)
    return dict(
        R0_rect=np.eye(4, dtype=np.float32),
        Tr_velo_to_cam=VELO_TO_CAM.copy(),
        P2=p2)


def _load_calib_offset(seq_dir, z_offset):
    calib_files = sorted((seq_dir / 'info_calib').glob('*.txt'))
    if not calib_files:
        return np.array([0.0, 0.0, z_offset], dtype=np.float32)
    with open(calib_files[0]) as f:
        lines = f.readlines()
    try:
        values = [float(v.strip()) for v in lines[1].split(',')]
        return np.array([values[1], values[2], z_offset], dtype=np.float32)
    except (IndexError, ValueError):
        return np.array([0.0, 0.0, z_offset], dtype=np.float32)


def _link_image(src_path, out_dir, sample_idx):
    if src_path is None:
        raise FileNotFoundError('Missing front camera frame for K-Radar.')
    dst_rel = Path('training') / 'image_2' / f'{sample_idx:06d}{src_path.suffix}'
    dst_path = out_dir / dst_rel
    if not dst_path.exists():
        try:
            os.symlink(src_path.resolve(), dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)
    image = mmcv.imread(str(dst_path))
    return image.shape, str(dst_rel)


def _find_frame_file(seq_dir, frame_id, dirs):
    for dirname in dirs:
        base = seq_dir / dirname
        if not base.exists():
            continue
        for ext in ('.mat', '.npy', '.jpg', '.jpeg', '.png'):
            exact = base / f'{frame_id}{ext}'
            if exact.exists():
                return exact
        for ext in ('.mat', '.npy', '.jpg', '.jpeg', '.png'):
            matches = sorted(
                path for path in base.glob(f'*{frame_id}*{ext}')
                if path.stem == frame_id or path.stem.endswith(f'_{frame_id}')
            )
            if matches:
                return matches[0]
    return None


def _read_split(path):
    if path is None:
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            keys.add((parts[0], Path(parts[1]).stem))
    return keys


def _sample_key(sample):
    return sample['seq'], sample['frame_id']
