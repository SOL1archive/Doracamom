"""Build a 6-digit KITTI-format tree under data/vod/training from VoD radar.

VoD ships ``radar_5frames`` with **5-digit** filenames (``00000.bin``) and
**.jpg** images, but mmdet3d's KITTI converter/dataset expect **6-digit** names
(``000000.bin``) and **.png** images. This creates a symlink tree that bridges
both (originals untouched):

    data/vod/training/{velodyne,calib,label_2}/{idx:06d}{.bin,.txt,.txt}
    data/vod/training/image_2/{idx:06d}.png  ->  .../{idx:05d}.jpg

Usage: python tools/prepare_vod_kitti.py
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

# (subdir, source ext, dest ext, zero-pad widths)
# velodyne needs BOTH widths: 6-digit for the converter's num_points step and
# 5-digit for KittiDataset._get_pts_filename ('{idx:05d}.bin') at runtime.
# image_2 / calib / label_2 are only read at 6-digit (converter + info paths).
SPECS = [('velodyne', '.bin', '.bin', (6, 5)),
         ('calib', '.txt', '.txt', (6,)),
         ('label_2', '.txt', '.txt', (6,)),
         ('image_2', '.jpg', '.png', (6,))]
ZERO12 = ' '.join(['0.0'] * 12)


def fix_calib_text(text):
    """VoD leaves ``Tr_imu_to_velo`` empty (radar->IMU not provided); fill it
    with zeros so mmdet3d's parser succeeds. The matrix is unused downstream
    (model/eval use Tr_velo_to_cam, R0_rect, P2)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out, has_imu = [], False
    for ln in lines:
        if ln.startswith('Tr_imu_to_velo'):
            has_imu = True
            if len(ln.split(':', 1)[1].split()) < 12:
                ln = f'Tr_imu_to_velo: {ZERO12}'
        out.append(ln)
    if not has_imu:
        out.append(f'Tr_imu_to_velo: {ZERO12}')
    return '\n'.join(out) + '\n'


def build(src_root, dst_root):
    src, dst = Path(src_root), Path(dst_root)
    if dst.is_symlink():
        dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for sub, src_ext, dst_ext, widths in SPECS:
        sdir = (src / sub).resolve()
        ddir = dst / sub
        if ddir.is_symlink():
            ddir.unlink()
        elif ddir.exists():
            shutil.rmtree(ddir)
        ddir.mkdir(parents=True)
        n = 0
        for f in sorted(glob.glob(str(sdir / f'*{src_ext}'))):
            idx = int(Path(f).stem)
            for width in widths:
                dpath = ddir / f'{idx:0{width}d}{dst_ext}'
                if sub == 'calib':
                    # rewrite as a real file with a valid Tr_imu_to_velo line
                    dpath.write_text(fix_calib_text(Path(f).read_text()))
                else:
                    os.symlink(os.path.abspath(f), dpath)
            n += 1
        print(f'{sub}: {n} src -> {ddir} (widths {widths}, {dst_ext})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--src',
        default='data/vod/view_of_delft_PUBLIC/radar_5frames/training')
    ap.add_argument('--dst', default='data/vod/training')
    args = ap.parse_args()
    build(args.src, args.dst)


if __name__ == '__main__':
    main()
