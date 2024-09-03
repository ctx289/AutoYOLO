""" basic rule parts """
import copy
import logging
import os

import numpy as np
from .rule_builder import RULE_PARTS


class BasicRules(object):
    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg

    def __call__(self, feed_dict, key, common_info=None):
        pass

