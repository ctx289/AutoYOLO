""" Base rules """
import json
import os

from ..builder import MODULES
from .rule_parts import build_rule_parts


@MODULES.register_module()
class BaseRules(object):
    def __init__(self,
                 rule_cfgs,
                 high_roi_path=None,
                 output_name='preds',
                 verbose=False,
                 gpu_id=0,
                 **kwargs):
        self.rule_cfgs = rule_cfgs
        self.high_roi_path = high_roi_path
        self.high_roi = None
        self.output_name = output_name
        self.verbose = verbose
        self.rule_funcs = list()
        self.rule_names = []
        self.gpu_id = gpu_id
        self.init(high_roi_path)

    def init(self, high_roi_path=None):
        """ init base rule modules

        Args:
            high_roi_path (str, optional): high roi path if has.
                if None, use default score other than adjusting from jitai
                Defaults to None.
        """

        # high roi config for jitai configration
        if high_roi_path is not None:
            self.high_roi = self.load_high_roi(high_roi_path)

        self.rule_names = []
        self.rule_funcs = list()
        for rule_cfg in self.rule_cfgs:
            self.rule_names.append(rule_cfg.type)
            func = build_rule_parts(rule_cfg)
            self.rule_funcs.append(func)

    def reset(self):
        """ reset cache and modules
        """
        self.init(self.high_roi_path)

    def load_high_roi(self, high_roi_path):
        """ Load high roi for jitai adjusting

        Args:
            high_roi_path (str): inside the path is a list of jsons with format
                Pxxx.json:
                    score_threshold: {LW: 0.1} # overall threshold for this pose
                    area_threshold: {YS: 0.6} # overall area threshold for this pose
                    attention_district: [
                        # box with new set score
                        {"box": [494, 2424, 962, 2835], "confidence": {"LW": 0.98}}
                    ]

        Returns:
            loaded high roi
        """
        high_roi = dict()
        poses = [x for x in os.listdir(high_roi_path) if x.startswith("P")]
        for pose_json in poses:
            pose = pose_json.split(".")[0]
            pose_json_path = os.path.join(high_roi_path, pose_json)
            with open(pose_json_path, "r") as f:
                pose_data = json.load(f)
            high_roi[int(pose[1:])] = pose_data
        return high_roi

    def update_high_roi(self, update_dict):
        """ update common info

        Args:
            update_dict (dict): key pose int, value is high roi value
                pose(int):
                    score_threshold: {LW: 0.1} # overall threshold for this pose
                    area_threshold: {YS: 0.6} # overall area threshold for this pose
                    attention_district: [
                        # box with new set score
                        {"box": [494, 2424, 962, 2835], "confidence": {"LW": 0.98}}
                    ]
        """
        if self.high_roi is None:
            self.high_roi = update_dict
        else:
            self.high_roi.update(update_dict)

    def get_common_info(self, feed_dict):
        common_info = {}
        pose = feed_dict['pose']
        # get high roi
        if self.high_roi is not None:
            gp_key = int(pose)
            try:
                high_roi = self.high_roi[gp_key]
            except KeyError:
                raise RuntimeError(
                    "Cannot find key {} in high config".format(gp_key))

            common_info['high_roi'] = high_roi
        return common_info

    def num(self):
        return len(self.rule_funcs)

    def __len__(self):
        return self.num()

    def __call__(self, feed_dict):
        '''
        from IPython import embed;embed()
        if self.output_name not in feed_dict.keys():
            return feed_dict
        '''
        common_info = self.get_common_info(feed_dict)
        rule_data = None
        for rule_func in self.rule_funcs:
            # Note:
            # A rule_fun must replace feed_dict[self.output_name] in the function
            # or return rule_data to be put in the feed_dict[self.output_name]
            rule_data = rule_func(feed_dict=feed_dict,
                                  key=self.output_name,
                                  common_info=common_info)

            # if rule func returns rule data, then put into feed dict
            if rule_data is not None:
                feed_dict[self.output_name] = rule_data
        '''
        # if rule_data is not None, then check filter key
        if feed_dict[self.output_name] is not None:
            for data in feed_dict[self.output_name]:
                if "filter" not in data.keys():
                    data["filter"] = 1
                    data['filter_by'] = "high_ng"
        '''
        return feed_dict

