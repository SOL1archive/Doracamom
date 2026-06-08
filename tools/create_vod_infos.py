"""Generate KITTI-format infos for the View-of-Delft (VoD) 4D-radar split.

The stock ``create_kitti_info_file`` hardcodes ``num_features=4`` in
``_calculate_num_points_in_gt``, which raises on VoD's **7-dim** radar points
whenever ``7 * num_points`` is not divisible by 4. This generator calls the same
mmdet3d helpers but with ``num_features=7`` (correct, and gives a correct
``num_points_in_gt``). It also skips the reduced-cloud / gt-database / 2D-anno
steps, which are unnecessary for evaluation + profiling and also assume 4-dim.

Usage:
    python tools/create_vod_infos.py --data-path data/vod --num-features 7
"""

import argparse
from pathlib import Path

import mmcv

from data_converter.kitti_converter import (_calculate_num_points_in_gt,
                                            _read_imageset_file,
                                            get_kitti_image_info)


def gen_split(data_path, split, num_features, training=True, label_info=True):
    ids = _read_imageset_file(str(Path(data_path) / 'ImageSets' / f'{split}.txt'))
    infos = get_kitti_image_info(
        str(data_path), training=training, label_info=label_info,
        velodyne=True, calib=True, image_ids=ids, relative_path=True)
    if label_info:
        _calculate_num_points_in_gt(
            str(data_path), infos, relative_path=True, num_features=num_features)
    out = Path(data_path) / f'kitti_infos_{split}.pkl'
    mmcv.dump(infos, out)
    print(f'{split}: {len(infos)} frames -> {out}')
    return infos


def main():
    parser = argparse.ArgumentParser(description='Create VoD KITTI infos.')
    parser.add_argument('--data-path', default='data/vod')
    parser.add_argument('--num-features', type=int, default=7)
    args = parser.parse_args()

    train = gen_split(args.data_path, 'train', args.num_features)
    val = gen_split(args.data_path, 'val', args.num_features)
    mmcv.dump(train + val, Path(args.data_path) / 'kitti_infos_trainval.pkl')
    print(f'trainval: {len(train) + len(val)} frames')


if __name__ == '__main__':
    main()
