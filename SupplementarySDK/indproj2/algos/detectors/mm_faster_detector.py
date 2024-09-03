import logging
import math
import os
from functools import partial

import cv2
import numpy as np

from ..builder import MODULES
from ..utils.error_code import error_code_dict
from .base_detector import BaseDetector


@MODULES.register_module()
class MMFasterDetector(BaseDetector):

    def __init__(
            self,
            config,
            ckpt,
            classes,
            keep_cats=None,
            poses=None,

            # nodes
            input_name='images',
            output_name='preds',

            # optimize and encrypt
            use_tiacc=False,
            optimize=False,         # deprecated
            encrypt=False,
            encrypt_platform=False, # deprecated in tiaoi 1.7.0
            optimize_input_shape=[{
                'seperate': '1*3*1280*1536'
            }],

            # roi
            roi_dir=None,
            roi_cfg=dict(type="LoadROISingle"),

            # preprocess
            rotate=None,
            crop=None,
            crop_overlap=50,
            crop_area=None,

            # postprocess
            min_wh=1,

            gpu_id=0,
            verbose=False,
            **kwargs):
        """ Init for mmdetection detection class
            other args please visit base_detector.py
        Args:
            optimize (bool, optional): whether use tiacc optimize. Defaults to False.
            use_tiacc (bool, optional): the same as optimize, but compatible
                with pingtai.
            optimize_input_shape (list, optional): input shape.
                Defaults to [{ 'seperate': '1*3*1280*1536' }].
            encrypt (bool, optional): whether encrypt
            encrypt_platform (bool, optional): whether decrpyt as platform format.
                Defaults to False.

            rotate (str, optional): mode to rotate, has two mode.
                must in [h>w, w>h]
                h>w: rotate to make sure iamge h > w
                w>h: rotate to make sure image w > h
                Defaults to None.
            crop (dict, optional): how to crop data
                if pose specified will crop othterwise will not
                dict(
                    1: 3, # if pose 1, crop to 3 images along long side
                    2: 2, # if pose 2, crop to 2 images along long side
                    3: 1  # if pose 3, crop to 1 images along long side
                )
            crop_overlap (int, optional): crop_overlap of crops

            min_wh (int, optional): min length of detected bbox side. default to 1

        """
        self.encrypt = encrypt
        self.encrypt_platform = encrypt_platform
        self.min_wh = min_wh
        super(MMFasterDetector, self).__init__(config=config,
                                               ckpt=ckpt,
                                               classes=classes,
                                               input_name=input_name,
                                               output_name=output_name,
                                               keep_cats=keep_cats,
                                               poses=poses,
                                               roi_dir=roi_dir,
                                               roi_cfg=roi_cfg,
                                               gpu_id=gpu_id,
                                               verbose=verbose)

        self.optimize = optimize
        self.use_tiacc = use_tiacc
        self.optimize_input_shape = optimize_input_shape
        self.opt_min_shape = None
        self.opt_max_shape = None

        if self.optimize and self.verbose:
            print("[Warning] Using optimize of acc is now deprecated \
                please use use_tiacc=True instead."                                                   )
        if self.optimize or self.use_tiacc:
            # tiacc optimize if sepcified
            self.model = self.tiacc_init(
                model=self.model,
                ckpt=self.ckpt,
                optimize_input_shape=self.optimize_input_shape)

        # rotate and crop
        assert rotate in [None, "h>w", "w>h"], \
            f"Unsupported rotate mode {rotate}"
        self.rotate = rotate
        self.crop = crop
        self.crop_area = crop_area
        self.crop_overlap = crop_overlap

    def tiacc_init(self, model, ckpt, optimize_input_shape=None):
        if self.verbose:
            logging.info("Converting model to tiacc")

        def get_shape(opt_shape):
            b, c, h, w = opt_shape.split("*")
            return (int(h), (int(w)))

        # define min max shape
        if isinstance(optimize_input_shape, list) \
                and isinstance(optimize_input_shape[0], dict) \
                and optimize_input_shape[0].get("range", None) is not None:

            self.opt_min_shape = get_shape(optimize_input_shape[0]['range'][0])
            self.opt_max_shape = get_shape(optimize_input_shape[0]['range'][1])

        import tiacc_inference

        opt_save_path = ckpt + ".opt"
        base_dir = os.path.dirname(opt_save_path)
        if not os.path.isdir(base_dir):
            opt_save_path = base_dir + ".opt"

        # if encrypt with platform, output opt as base dir
        if self.encrypt_platform:
            from ..utils.encrypt import get_platform_asset_path
            opt_save_path = get_platform_asset_path(ckpt)
            opt_save_path = opt_save_path.rstrip("/") + ".opt"

        if os.path.exists(opt_save_path):
            if self.verbose:
                logging.info("use tiacc load mode")
            opt_model, report = tiacc_inference.load(model,
                                                     load_path=opt_save_path)
        else:
            if self.verbose:
                print("use tiacc save mode")
            opt_model, report = tiacc_inference.optimize(
                model,
                1,
                0,
                input_shapes=optimize_input_shape,
                save_path=opt_save_path)
        return opt_model

    def _set_device(self, device_type: str, device_id: int = 0) -> None:
        """
        for sdk _set_device
        """
        print(f'set MMFasterDetector device to {device_type}:{device_id}')
        self.model.to(f'{device_type}:{device_id}')
        self.gpu_id = device_id

    @property
    def with_acc(self):
        return self.optimize or self.use_tiacc

    def _precheck(self, image, feed_dict):
        """ precheck input
        """
        status = True
        reason = "Success."

        # 1. Input shape check for Tiacc Range (Now deprecated because
        #   it is unreasonable to check in pipeline code, need to check in mmdet/api/inference.py
        #   or in tiacc code)
        # if self.with_acc and self.opt_min_shape is not None:
        #     if image.shape[0] > self.opt_max_shape[0] - 32 or image.shape[
        #             1] > self.opt_max_shape[1] - 32:
        #         status = False
        #         reason = f"Unmatch shape for tiacc, got {image.shape} > {self.opt_max_shape}"
        #     if image.shape[0] < (self.opt_min_shape[0] + 32) or image.shape[
        #             1] < (self.opt_min_shape[1] + 32):
        #         status = False
        #         reason = f"Unmatch shape for tiacc, got {image.shape} < {self.opt_min_shape}"

        if not status:
            feed_dict['error_code'] = error_code_dict['detection_fail']
            feed_dict['error_reason'] = reason

        return status

    def init(self, config, ckpt, device, import_custom_modules=True):
        """ Init funciton for detector

        Including several modes
        1. decrypt
            normal encrypt would be two level folder, such as
                - xxx
                    - xxx.config
                    - xxx.pth
            platform encrypt is special, many have several depth as below
                - all_model(Name can be varied)
                    - Pretrained
                    - Detection
                        - config.py
                        - latest.pth
        2. init

        Args:
            config (str): config path
            ckpt (str): ckpt path
            device (str): device for current process
            import_custom_modules (bool, optional): custom funciton in mmdet
                config. Defaults to True.

        Returns:
            initialized detector and may be tiacc capsulated later
        """
        from mmdet.apis import inference_detector, init_detector
        if self.verbose:
            logging.info("Initializing MMFasterDetector on {}".format(device))
        self.inference_detector = inference_detector

        if self.encrypt:
            config, ckpt = self._encrypt_bilevel(config, ckpt)
        elif self.encrypt_platform:
            config, ckpt = self._encrypt_platform(config, ckpt)

        if (self.encrypt or self.encrypt_platform
            ) and import_custom_modules and config.get('custom_imports', None):
            from mmcv.utils.misc import import_modules_from_strings
            import_modules_from_strings(**config['custom_imports'])

        return init_detector(config, ckpt, device=device)

    def _inference(self, image):
        """ get mmdet inference results with format

        Args:
            image (np.array)
        """

        results = self.inference_detector(self.model, image)
        # if results is tuple, take results 0
        if isinstance(results, tuple):
            results = results[0]

        bboxes = np.vstack(results)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(results)
        ]
        labels = np.concatenate(labels)
        pred_list = []
        for bbox, label in zip(bboxes, labels):
            # pesudo segms
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            if (xmin >= xmax - self.min_wh) or (ymin >= ymax - self.min_wh):
                continue
            area = (ymax - ymin) * (xmax - xmin)
            length=np.sqrt((ymax - ymin) ** 2 + (xmax - xmin) ** 2)
            boundary = None

            # keep cats if needed
            if self.keep_cats is not None and self.classes[
                    label] not in self.keep_cats:
                continue

            pred_list.append({
                "det_score": float(bbox[-1]),
                "det_bbox": bbox[:-1].astype(np.int).tolist(),
                "det_code": self.classes[label],
                "area": int(area),
                "length": int(length),
                "polygon": boundary,
            })
        return pred_list

    def _rotate_image(self, image, image_meta):
        """rotate imgae colockwse 90 degree to align image all vertical or
            hrizontal

            all image has same h or w as longer side
            ++++      ++++++++
            +  +  =>  +      +
            +  +      ++++++++
            ++++
        """

        def _bbox_back(bbox, image_shape, mode="ROTATE_90_CLOCKWISE"):
            if mode == "ROTATE_90_CLOCKWISE":
                x1, y1, x2, y2 = bbox
                h, w = image_shape[:2]
                new_bbox = [y1, h - x2, y2, h - x1]
            else:
                raise NotImplementedError(
                    f"Unimplemented rotate for bbox {mode}")
            return new_bbox

        if (image.shape[0] > image.shape[1] and self.rotate == 'w>h') \
                or (image.shape[1] > image.shape[0] and self.rotate == 'h>w'):
            image_rotate = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
            image_meta['rotate'] = partial(_bbox_back,
                                           image_shape=image.shape,
                                           mode="ROTATE_90_CLOCKWISE")

            return image_rotate, image_meta
        return image, image_meta

    @property
    def with_crop(self):
        return self.crop is not None or self.crop_area is not None

    def _crop_image(self, image, image_meta, pose):
        """crop image to several pices with overlap
        Current cropping has two mode
            1. by longer side to several pices in self.crop
                self.crop:
                    pose1: 2 # pose1 crop 2 pices
                    pose2: 3 # pose2 crop 3 pices
            2. by area, each pose will crop num area/longer_side_lenght
        """

        # Function for transforming bbox back
        def _bbox_back(bbox, crop_from, on_h=True):
            x1, y1, x2, y2 = bbox
            if on_h:
                return [x1, y1 + crop_from, x2, y2 + crop_from]
            else:
                return [x1 + crop_from, y1, x2 + crop_from, y2]

        # Function for cropping image along one side
        def _crop_img(image, cropped_images, image_meta, crop_from, crop_to,
                      on_h):
            image_meta['crop'].append(
                partial(_bbox_back, crop_from=crop_from, on_h=on_h))
            if on_h:
                cropped_images.append(image[crop_from:crop_to])
            else:
                cropped_images.append(image[:, crop_from:crop_to])

        # if the pose is not set or not crop, return the whole image
        if (self.crop is not None and self.crop.get(pose, None) is None):
            image_meta['crop'] = [partial(_bbox_back, crop_from=0, on_h=True)]
            return [image], image_meta

        # Check crop by longer side or by area and calculate num of crops
        if self.crop is not None:
            crop_num = self.crop[pose]
        else:
            min_shape = min(image.shape[:2])
            max_shape = max(image.shape[:2])
            crop_num = math.ceil(max_shape / self.crop_area * min_shape)
        on_h = image.shape[0] > image.shape[1]
        length = image.shape[0] if on_h else image.shape[1]
        crop_length = length // crop_num

        # Start cropping
        cropped_images = []
        image_meta['crop'] = []
        for idx in range(crop_num):
            crop_from = crop_length * idx - self.crop_overlap

            # first crop
            if idx == 0:
                _crop_img(image,
                          cropped_images=cropped_images,
                          image_meta=image_meta,
                          crop_from=0,
                          crop_to=crop_length + self.crop_overlap,
                          on_h=on_h)

            # last crop
            elif idx == crop_num - 1:
                _crop_img(image,
                          cropped_images=cropped_images,
                          image_meta=image_meta,
                          crop_from=crop_from,
                          crop_to=length,
                          on_h=on_h)

            else:
                # normal crop
                if on_h:
                    crop_to = min(crop_length * (idx + 1) + self.crop_overlap,
                                  image.shape[0])
                else:
                    crop_to = min(crop_length * (idx + 1) + self.crop_overlap,
                                  image.shape[1])
                _crop_img(image,
                          cropped_images=cropped_images,
                          image_meta=image_meta,
                          crop_from=crop_from,
                          crop_to=crop_to,
                          on_h=on_h)

        return cropped_images, image_meta

    def _transform_back(self, pred_list, image_meta):

        if "crop" in image_meta.keys():
            new_pred_list = []
            for pred_idx, preds in enumerate(pred_list):
                for pred in preds:
                    pred['det_bbox'] = image_meta['crop'][pred_idx](
                        bbox=pred['det_bbox'])
                    new_pred_list.append(pred)
            pred_list = new_pred_list

        if "rotate" in image_meta.keys():
            for pred in pred_list:
                pred['det_bbox'] = image_meta['rotate'](bbox=pred['det_bbox'])
        return pred_list

    def predict(self, feed_dict, **kwargs):

        images = feed_dict[self.input_name]
        # mmdetector only support on image inference
        if isinstance(images, (list, tuple)):
            image = images[0]
        else:
            image = images

        image_meta = dict()
        if self.rotate is not None:
            image, image_meta = self._rotate_image(image, image_meta)

        if self.with_crop:
            image_crops, image_meta = self._crop_image(image,
                                                       image_meta=image_meta,
                                                       pose=feed_dict['pose'])
            pred_list = []
            for img in image_crops:
                # check input before inference
                if not self._precheck(img, feed_dict):
                    pred_list.append([])
                else:
                    pred_list.append(self._inference(img))
        else:
            # check input before inference
            if not self._precheck(image, feed_dict):
                pred_list = []
            else:
                pred_list = self._inference(image)

        # transform predictions back if needed
        pred_list = self._transform_back(pred_list, image_meta)

        self.feed_data(feed_dict, self.output_name, pred_list)
        return feed_dict
