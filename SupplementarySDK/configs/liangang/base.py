block_pose=set(range(1, 17)) - set([15])

############# pipeline config ##################
pipeline = dict(
    type="BasePipeline",
    modules=[
        dict(
            type="BaseRules",
            rule_cfgs=[
                dict(type="FilterBySizeMeasurement", thresh=57),
            ])
    ])
