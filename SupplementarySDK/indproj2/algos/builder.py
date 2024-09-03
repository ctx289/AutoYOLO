""" model builder for whole pipeline """
from mmcv.utils import Registry, build_from_cfg

PIPELINE = Registry('pipelines', scope='Youtu')
MODULES = Registry('modules', scope='Youtu')

registry_dict = {
    'pipeline': PIPELINE,
}


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    return build_from_cfg(cfg, registry, default_args)


def build_module(cfg, default_args=None):
    return build(cfg, MODULES, default_args)


def build_pipeline(cfg, gpu_id=0, **default_args):
    """Build pipeline."""
    default_args['gpu_id'] = gpu_id
    return build(cfg, PIPELINE, default_args=default_args)
