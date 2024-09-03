block_pose = []
pipeline = dict(
    type='BasePipeline',
    modules=[
        dict(
            type='UltralyticsFasterDetector',
            config='/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_transform_yz_bottlecap1/train/args.yaml',
            ckpt='/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_transform_yz_bottlecap1/train/weights/best.pt',
            classes=["YZPG1_ZW",],
            encrypt_platform=False,
            encrypt=False,
            crop_by_seg=False,
            crop_by_outer=False,)
    ])
