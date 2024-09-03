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
            type='UltralyticsFasterDetector',
            config=None,
            ckpt='$IND_MODEL_PATH2/weights/best.pt',
            classes=None,
            crop_by_seg=True)
    ])
