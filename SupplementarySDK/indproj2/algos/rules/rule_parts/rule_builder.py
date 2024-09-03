""" rules """
from mmcv.utils import Registry, build_from_cfg

RULE_PARTS = Registry('rule_parts', scope='Youtu')


def build_rule_parts(rule_part_cfg):
    return build_from_cfg(rule_part_cfg, RULE_PARTS)
