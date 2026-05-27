_base_ = ['./Doracamom_TJ4D.py']

data_root = 'data/KRadar/'
point_cloud_range = [0, -16.0, -2.0, 72.0, 16.0, 7.6]
voxel_size = [0.16, 0.16, 9.6]
occ_size = [180, 80, 24]
class_names = ['Car', 'Truck']
od_num_classes = 2
bev_h_ = 80
bev_w_ = 180
bev_z_ = 9
img_scale = (1280, 960)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0, 1, 2, 3, 5]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=img_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0, 1, 2, 3, 5]),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=img_scale, multiscale_mode='value', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'points'])
        ])
]

model = dict(
    bev_h=bev_h_ * 2,
    bev_w=bev_w_ * 2,
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size),
    pts_voxel_encoder=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size),
    pts_middle_encoder=dict(output_shape=[200, 450]),
    pts_bbox_head=dict(
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        od_num_classes=od_num_classes,
        transformer=dict(
            cam_encoder=dict(
                pc_range=point_cloud_range,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='OccSpatialAttentionV1',
                            pc_range=point_cloud_range,
                            num_cams=1,
                            deformable_attention=dict(
                                type='MSDeformableAttention3DV1',
                                embed_dims=256,
                                num_points=bev_z_,
                                num_levels=4),
                            embed_dims=256)
                    ])),
            temporal_encoder=dict(bev_h=bev_h_, bev_w=bev_w_, bev_z=bev_z_),
            voxel_decoder=dict(occ_size=occ_size, bev_h=bev_h_, bev_w=bev_w_, bev_z=bev_z_),
            seg_decoder=dict(num_classes=12),
            occbevfusion2d=dict(img_bev_conv_channel=576)),
        decoder_cfg=dict(
            num_classes=od_num_classes,
            anchor_generator=dict(
                ranges=[
                    [0, -16.0, -1.08, 72.0, 16.0, -1.08],
                    [0, -16.0, -1.08, 72.0, 16.0, -1.08],
                ],
                sizes=[[2.1, 4.2, 2.0], [3.2, 9.5, 3.7]]),
            train_cfg=dict(
                assigner=[
                    dict(
                        type='MaxIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.2,
                        min_pos_iou=0.2,
                        ignore_iof_thr=-1),
                    dict(
                        type='MaxIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.45,
                        neg_iou_thr=0.15,
                        min_pos_iou=0.15,
                        ignore_iof_thr=-1),
                ]))))

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'kradar_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'kradar_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'kradar_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names))

evaluation = dict(interval=1, pipeline=test_pipeline)
work_dir = 'work_dirs/Doracamom_KRadar'
load_pts_from = None
load_from = None
resume_from = None
