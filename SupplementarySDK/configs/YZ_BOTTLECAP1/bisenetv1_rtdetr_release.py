block_pose = []
pipeline = dict(
    type='BasePipeline',
    modules=[
        dict(
            type='MMRoiSegmentor',
            config=
            '$IND_MODEL_PATH2/bisenetv1_r18-d32_in1k-pre_4x4_20k.py/bisenetv1_r18-d32_in1k-pre_4x4_20k.py',
            ckpt=
            '$IND_MODEL_PATH2/bisenetv1_r18-d32_in1k-pre_4x4_20k.py/iter_20000.pth',
            classes=dict({
                0: '_background_',
                1: 'POLYGON'
            }),
            get_polygon=True),
        dict(
            type='PPFasterDetector',
            config='$IND_MODEL_PATH2/rtdetr_r18vd_6x_online_crop/infer_cfg.yml',
            ckpt='$IND_MODEL_PATH2/rtdetr_r18vd_6x_online_crop/',
            classes=['YZPG1_ZW'],
            crop_by_seg=True)
    ])
