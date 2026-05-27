_base_ = ['./Doracamom_TJ4D.py']

data_root = 'data/KRadar/'

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'kradar_infos_train.pkl'),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'kradar_infos_val.pkl'),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'kradar_infos_val.pkl'))

work_dir = 'work_dirs/Doracamom_KRadar'
load_pts_from = None
load_from = None
resume_from = None
