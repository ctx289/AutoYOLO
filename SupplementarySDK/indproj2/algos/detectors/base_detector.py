""" base detector for 2d detection """
from abc import abstractclassmethod

from ..builder import build_module


class BaseDetector(object):

    def __init__(self,
                 config,
                 ckpt,
                 classes,
                 input_name='images',
                 output_name='preds',
                 keep_cats=None,
                 poses=None,
                 roi_dir=None,
                 roi_cfg=dict(type="LoadROISingle"),
                 gpu_id=0,
                 verbose=False):
        """BaseDetector for detection functions

        Args:
            config (str): path to model config
            ckpt (str): path to model ckpt
            classes (list): list of classes to convert from category_id to str
            input_name (str, optional): the input node name.
                Defaults to 'images'.
            output_name (str, optional): the output node name.
                Defaults to 'preds'.
            keep_cats (list, optional): list of cats to keep,
                if keep all set to None. Defaults to None.
            poses (list, optional): list of int for detection poses,
                such as [1]: means only detect pose 1. Defaults to None.
            roi_dir (str, optional): path to roi, if set will check each box
                in roi or not. Defaults to None.
            roi_cfg (dict, optional): information to build roi if roi_dir is
                not None. Defaults to dict(type="LoadROISingle").
                Full config please visit indproj/algos/others/load_roi.py
                such as below:
                dict(type="LoadROISingle", # LoadROI
                    block_pose=[],
                    roi_dilate=50)
            gpu_id (int, optional): gpu id. Defaults to 0.
            verbose (bool, optional): whether to print debug information.
                Defaults to False.
        """
        self.config = config
        self.ckpt = ckpt
        self.classes = classes
        self.input_name = input_name
        self.output_name = output_name
        self.keep_cats = keep_cats
        self.poses = poses
        self.verbose = verbose
        self.gpu_id = gpu_id

        self.roi_cfg = roi_cfg
        self.roi_dir = roi_dir
        # deprecated
        # if self.roi_dir is not None:
        #     roi_cfg['roi_dir'] = roi_dir
        #     self.roi_filter = build_module(roi_cfg)

        self.model = self.init(
            self.config, self.ckpt, device=f"cuda:{self.gpu_id}")

    @abstractclassmethod
    def init(self, config, ckpt, device):
        pass

    @abstractclassmethod
    def predict(self, feed_dict):
        pass

    # @property
    # def with_roi(self):
    #     return self.roi_dir is not None

    def feed_data(self, feed_dict, key, value):
        try:
            feed_dict[key].extend(value)
        except KeyError:
            feed_dict[key] = []
            feed_dict[key].extend(value)
        return feed_dict

    def __call__(self, feed_dict, **kwargs):
        """ inference with feed data

        Args:
            feed_dict (dict)
        """
        if self.poses is not None and feed_dict['pose'] not in self.poses:
            if self.output_name not in feed_dict:
                feed_dict[self.output_name] = []
        else:
            feed_dict = self.predict(feed_dict)

        # if self.with_roi:
        #     feed_dict = self.roi_filter(feed_dict)

        return feed_dict
