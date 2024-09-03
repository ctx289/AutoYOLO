#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
import os
import shutil
import time
import traceback

import cv2
import numpy as np
import requests
from mmcv import Config

from indproj2.algos.utils.error_code import error_code_dict
from indproj2.algos.utils.pose_utils import get_group, get_pose
from indproj2.apis.inference import init_pipeline
from utils2.config_utils import fix_dict, get_default_config_path


class AlgorithmInterface:

    def __init__(self, gpu_id=0, invoke_from_train_tool=False, verbose=True, config_path=None, check_license=False):
        """ algorithm interface

        Args:
            gpu_id (int, optional): gpu id. Defaults to 0.
            verbose (bool, optional): whether output infos. Defaults to True.
            config_path (str, optional): config for the project. Defaults to None.
        """
        if not invoke_from_train_tool and check_license:
            # request license
            header = {"Content-Type": "application/json"}
            json = {}
            register_url = "http://127.0.0.1:9097/license/GetLicenseInfo"
            # 发送post请求
            try_cnt = 24
            for try_idx in range(try_cnt):
                try:
                    response = requests.post(url=register_url, json=json, headers=header, timeout=5)
                    license_status = response.json()["LicenseData"]["Status"]
                    if license_status != 1:
                        print("license error, return: ", license_status)
                        self.init_code = -1
                        return
                    else:
                        print("license ok")
                        break
                except Exception as e:
                    print(f"{time.time()}, license exception: {traceback.format_exc()}")
                    time.sleep(5)

                    if try_idx == try_cnt - 1:
                        self.init_code = -1
                        return

        self.init_code = -1000
        self.gpu_id = gpu_id
        self.verbose = verbose

        if config_path is None:
            self.config_path = get_default_config_path()
        else:
            self.config_path = config_path
        self.config = Config.fromfile(self.config_path)

        self.config = fix_dict(self.config)

        self.pipeline = init_pipeline(self.config,
                                      self.gpu_id,
                                      verbose=self.verbose)
        self.sample_code_priority = self.config.get("sample_code_priority", [])
        self.init_code = 0

    def get_init_code(self):
        return self.init_code

    def with_module(self, name_type):
        return self.pipeline.with_module(name_type)

    def GetCurrentSdk(self):
        return {'error_code': 0, 'error_msg': "success", "result": {}}

    def ReloadStrategyCfg(self, path):
        "Only reset modules based on whether a module has cache to reset"
        self.pipeline.reset()
        return {"error_code": 0, "error_msg": "OK"}

    def process(self, feed_dict, pose, group, time_statistic=False):
        if 'images' not in feed_dict:
            raise ValueError("Incomplete `feed_dict`, `images` is missing!")

        # prepare feed_dict
        feed_dict.update(dict(pose=pose, group=group))
        feed_dict['error_code'] = error_code_dict['success']
        feed_dict['error_reason'] = "success"

        # run pipeline
        if time_statistic:
            results, time_dict = self.pipeline.time_call(feed_dict=feed_dict)
            return results, time_dict
        else:
            results = self.pipeline(feed_dict=feed_dict)
            return results, {}

    def _add_basic_info(self, box, pose, feed_dict):
        res = {}
        res["bbox"] = box["det_bbox"]
        res["code"] = box["det_code"] if "det_code_convert" not in box.keys(
        ) else box['det_code_convert']
        res['polygon'] = box['polygon']
        res["score"] = float(box["det_score_second"]
                             ) if 'det_score_second' in box.keys() else float(
            box["det_score"])
        res["segmentation"] = box["polygon"]
        res["area"] = int(box["area"])
        res["length"] = int(box["length"])

        if "real_area" not in box.keys():
            box['real_area'] = 0.0
        res['real_area'] = box['real_area']

        return res

    def _clear_useless_data(self, feed_dict):
        try:
            feed_dict.pop("images")
        except KeyError:
            pass
        # pop roi_mask and roi_polygon if needed
        if 'roi_seg' in feed_dict.keys():
            feed_dict['roi_seg'].pop('roi_mask')
            feed_dict['roi_seg'].pop('roi_polygon')

    def _get_default_response(self, req, width, height, pose, group, time_dict):
        resp = {}
        resp["image_width"] = width
        resp["image_height"] = height
        resp["imageName"] = req['imageName']
        resp["timestamp"] = time.time()
        resp["time_stage"] = time_dict
        resp["pose"] = pose
        resp["group"] = group
        resp["result"] = []
        resp["origin_info"] = []
        resp["error_code"] = 0
        resp["error_msg"] = "success"
        return resp

    def SinglePosePredict(self, req, time_statistic=False):
        """response

        Args:
            req (dict): input request with imageData and imageName
            time_statistic (bool): print time statustics

        Raises:
            ValueError: if imageData is not found

        Returns:
            a dict
            interface output basic format as follow
                res: [
                    bbox: list
                    code: str
                    score: float
                    polygon: list
                    segmentation: same as polygon
                    area:: float
                    real_area: float
                    calib_ok: str if has
                    calibration: matrix if has
                ] # list of returned ng info
                pose: int
                group: int
                image_width: str
                image_height: str
                imageName: str
                timestamp: timestamp
                error_code: int
                error_reason: str
                origin_info: dict only for internal usage

        """
        req = req["images"][0]
        req["imageData"] = req["image_data"]
        req["imageName"] = req["image_name"]

        # if iamgeData does not exist
        if 'imageData' not in req:
            raise ValueError("Incomplete `request`, `imageData` is missing!")

        # read request
        feed_dict = dict(images=[req["imageData"]], image_name=req["imageName"])

        pose = get_pose(req["imageName"], None)
        group = get_group(req["imageName"], None)

        # get default response
        height, width = feed_dict['images'][0].shape[:2]
        resp = self._get_default_response(
            req=req,
            width=width,
            height=height,
            pose=pose,
            group=group,
            time_dict=None)
        
        # block pose
        if pose in self.config.get("block_pose", []):
            self._clear_useless_data(feed_dict)
            resp["origin_info"] = feed_dict
            return resp
        
        # predict
        results, time_dict = self.process(feed_dict,
                                          pose,
                                          group,
                                          time_statistic=time_statistic)
        resp["time_stage"] = time_dict

        # construct response
        resp["error_code"] = results['error_code']
        resp["error_msg"] = results['error_reason']

        # get results
        # if no detection ############
        if "preds" not in results.keys():
            self._clear_useless_data(feed_dict)
            resp['result'] = []
            resp["origin_info"] = feed_dict
            return resp
        
        # if has detections ###########
        boxes = results["preds"]
        filtered_boxes = [x for x in boxes if x.get("filter", 1)]
        for box in filtered_boxes:
            res = self._add_basic_info(box, pose, results)
            resp["result"].append(res)

        self._clear_useless_data(feed_dict)
        resp["origin_info"] = feed_dict
        return resp

    # for engineer team
    def SamplePostProcess(self, results):
        """ get sample result

        Args:
            feed_dict (list): the element of the list is
                the resp returned from response, format see function response

        Returns:
            # if OK and calib wrong:
            #     dict(
            #         code='UNKNOWN',
            #         score=1.0
            #     )
            if OK
                dict(
                    code='OK',
                    score=1.0
                )
            if not OK:
                dict(
                    code=NG_CODE,
                    score=score_for_ng
                )
        """
        stat_dict = dict(code='OK', score=1.0)

        ## check calib fail status
        # has_calib_fail = False
        # for result in results:
        #     if result['error_code'] == error_code_dict['calibration_fail']:
        #         has_calib_fail = True

        # use simple rules
        ng_codes = [x['code'] for y in results for x in y['result']]
        ng_scores = [x['score'] for y in results for x in y['result']]

        # Judging Rule for ok/ng/calib-failed
        # if OK sample (no mark and no ng_code mean the sample ok)
        #   if calib failed
        #       output UNKNOWN, because the sample may have big defect and cannot be calibrated
        #   else:
        #       output OK
        # if NG sample:
        #   return NG
        if not len(ng_codes):
            # if has_calib_fail:
            #     stat_dict['code'] = 'UNKNOWN'
            return stat_dict

        # if NG sample, output NG
        if self.sample_code_priority:
            ng_codes_set = set(ng_codes)
            for code in self.sample_code_priority:
                if code in ng_codes_set:
                    stat_dict['code'] = code
                    stat_dict['score'] = max([
                        x['score'] for y in results for x in y['result']
                        if x['code'] == code
                    ])
                    return stat_dict

        stat_dict['code'] = ng_codes[np.argmax(ng_scores)]
        stat_dict['score'] = np.max(ng_scores)
        return stat_dict


if __name__ == "__main__":
    """
    # 待测点位
    python3 ./interface2/algorithm_interface_standard.py ./configs/liangang/base.py /youtu/xlab-team4/ryanwfu/24_second_sdk/625_dev/NG/S00040_C15_P15_L0_PI139_G1_M1_Y1_20230628200816.jpg
    python3 ./interface2/algorithm_interface_standard.py ./configs/liangang/base.py /youtu/xlab-team4/ryanwfu/24_second_sdk/625_dev/OK/S00003_C15_P15_L0_PI139_G1_M1_Y1_20230628200051.jpg
    # 其他点位
    python3 ./interface2/algorithm_interface_standard.py ./configs/liangang/base.py /youtu/xlab-team4/ryanwfu/24_second_sdk/625_dev/other_pose/S00016_C15_P13_L0_PI139_G1_M1_Y1_20230628200318.jpg
    # YZ_BOTTLECAP1
    python3 ./interface2/algorithm_interface_standard.py ./configs/YZ_BOTTLECAP1/base.py /youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/val/NG/S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png
    """
    import sys
    import cv2

    interface = AlgorithmInterface(config_path=sys.argv[1], check_license=False)

    # test inference
    image_path = sys.argv[2]
    image_data = cv2.imread(image_path)

    req = {"images": [{"image_name": image_path, "image_data": image_data}]}
    resp = interface.SinglePosePredict(req, time_statistic=True)
    print(resp)

    # test speed
    # for i in range(100):
    #     resp = interface.SinglePosePredict(req, time_statistic=True)
    #     print(resp['time_stage'])

