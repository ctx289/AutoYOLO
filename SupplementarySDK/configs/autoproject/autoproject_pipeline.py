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
            type="PPFasterDetector",
            config="/youtu/xlab-team4/ryanwfu/26_AutoModels/OUTPUT/task_YZ_BOTTLECAP1/rtdetr_r18vd_6x_bottle_cap_online_crop.yml/output_inference/rtdetr_r18vd_6x_bottle_cap_online_crop/infer_cfg.yml",
            ckpt="/youtu/xlab-team4/ryanwfu/26_AutoModels/OUTPUT/task_YZ_BOTTLECAP1/rtdetr_r18vd_6x_bottle_cap_online_crop.yml/output_inference/rtdetr_r18vd_6x_bottle_cap_online_crop/",
            classes=["YZPG1_ZW"],
            crop_by_seg=True,
        )
    ])
