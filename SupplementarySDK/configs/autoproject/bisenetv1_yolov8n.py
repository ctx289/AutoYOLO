block_pose=[]

############# pipeline config ##################
pipeline = dict(
    type="BasePipeline",
    modules=[
        dict(
            type="MMRoiSegmentor",
            config="/youtu/xlab-team4/ryanwfu/26_AutoModels/OUTPUT/task_YZ_BOTTLECAP1/bisenetv1_r18-d32_in1k-pre_4x4_20k.py/bisenetv1_r18-d32_in1k-pre_4x4_20k.py",
            ckpt="/youtu/xlab-team4/ryanwfu/26_AutoModels/OUTPUT/task_YZ_BOTTLECAP1/bisenetv1_r18-d32_in1k-pre_4x4_20k.py/iter_8000.pth",
            classes={0:'_background_', 1:'POLYGON'},
            get_polygon=True,
        ),
        dict(
            type="UltralyticsFasterDetector",
            config=None,
            ckpt="/youtu/xlab-team4/ryanwfu/26_AutoModels/OUTPUT/task_YZ_BOTTLECAP1/yolov8n_sgd_12b_500e_800_800_copypaste/weights/best.pt",
            classes=None,
            crop_by_seg=True,
        )
    ])
