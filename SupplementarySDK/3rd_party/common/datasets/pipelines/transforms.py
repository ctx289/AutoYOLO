import copy
import random

import re, os
import cv2
import mmcv
import numpy as np
from mmcv.utils import Registry
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import RandomCrop, Resize

COMMON_PIPELINES = Registry("pipeline", scope="common", parent=PIPELINES)

@COMMON_PIPELINES.register_module()
class ResizeByRatio(Resize):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',
                 test_time_keep_ratio=False,
                 load_img_shape=False,
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            # assert len(self.img_scale) == 1
            # NOTE. pass @choasliu
            pass
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

        # NOTE. test time keep ratio, not use absolute range.
        # 'False' use img_scale, 'True' use image_scale_area * ratio
        self.load_img_shape = load_img_shape
        self.test_time_keep_ratio = test_time_keep_ratio

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        # NOTE for training-time, if self.img_scale is None, use image shape @choasliu
        if self.load_img_shape == True:
            ratio = np.sqrt((self.img_scale[0][0] * self.img_scale[0][1]) / (results['img'].shape[0] * results['img'].shape[1]))
            self.img_scale = [(int(results['img'].shape[0] * ratio), int(results['img'].shape[1] * ratio))]
        
        # NOTE for test-time, keep ratio @choasliu
        if self.test_time_keep_ratio == True and 'scale' in results:
            ratio = np.sqrt((results['scale'][0] * results['scale'][1]) / (results['img'].shape[0] * results['img'].shape[1]))
            results['scale'] = (int(results['img'].shape[0] * ratio), int(results['img'].shape[1] * ratio))

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@COMMON_PIPELINES.register_module()
class ResizeByPose(Resize):
    def __init__(self,
                 ratio_range_dict={},
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.ratio_range_dict = ratio_range_dict
    
    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        image_name = os.path.basename(results['filename'])
        pose = int(re.search(r"P(\d+)", image_name).group(1))
        ratio_range = self.ratio_range_dict[pose] if pose in self.ratio_range_dict else self.ratio_range

        if ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx


@COMMON_PIPELINES.register_module()
class Mosaic(object):
    def __init__(
        self, prob=0.5, mosaic_center_ratio=0.5, no_empty_buffer=True, buffer_size=10
    ):
        """mosaic method in yolo but less random

        Args:
            prob (float, optional): prob to use mosaic. Defaults to 0.5.
            mosaic_center_ratio (float, optional): how much area not to
                    use when random center. Defaults to 0.5.
            no_empty_buffer (bool, optional): whether not buffer empty image.
                    Defaults to True.
            buffer_size (int, optional): buffer size, the bigger the more
                    ratio of random. Defaults to 10.
        """
        self.prob = prob
        self.mosaic_center_ratio = mosaic_center_ratio
        self.results_buffer = []
        self.no_empty_buffer = no_empty_buffer
        self.buffer_size = buffer_size

    def mosaic_main(self, results):
        # back up results
        use_this_idx = np.random.randint(4)

        # set compose base image
        this_img = results["img"]
        height, width, channels = this_img.shape[:3]
        img4 = np.zeros((2 * height, 2 * width, channels), dtype=this_img.dtype)

        # set image center
        xc = int(
            np.random.uniform(
                width * self.mosaic_center_ratio, width * (2 - self.mosaic_center_ratio)
            )
        )
        yc = int(
            np.random.uniform(
                height * self.mosaic_center_ratio,
                height * (2 - self.mosaic_center_ratio),
            )
        )

        # start form img and label
        labels4 = []
        bboxes4 = []
        h = height
        w = width
        current_result = None
        for idx in range(4):
            if idx == use_this_idx:
                current_result = results
            else:
                current_result = self.results_buffer[
                    np.random.randint(self.buffer_size)
                ]

            current_img = current_result["img"]
            min_w = current_img.shape[1]
            min_h = current_img.shape[0]
            if current_img.shape[0] < h or current_img.shape[1] < w:
                new_img = np.zeros((height, width, channels), dtype=this_img.dtype)
                min_w = min(new_img.shape[1], current_img.shape[1])
                min_h = min(new_img.shape[0], current_img.shape[0])
                new_img[:min_h, :min_w] = current_img[:min_h, :min_w]
                current_img = new_img

            if idx == 0:  # top left
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w * 2), min(h * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # img4[ymin:ymax, xmin:xmax]
            img4[y1a:y2a, x1a:x2a] = current_img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if (
                "gt_bboxes" in current_result.keys()
                and len(current_result["gt_bboxes"]) > 0
            ):
                gt_bboxes = (
                    np.array(current_result["gt_bboxes"]).copy().reshape([-1, 4])
                )

                gt_bboxes[:, 0] = gt_bboxes[:, 0] + padw
                gt_bboxes[:, 1] = gt_bboxes[:, 1] + padh
                gt_bboxes[:, 2] = gt_bboxes[:, 2] + padw
                gt_bboxes[:, 3] = gt_bboxes[:, 3] + padh
                gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], x1a, x2a)
                gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], y1a, y2a)
                select_idx = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                    gt_bboxes[:, 3] > gt_bboxes[:, 1]
                )

                labels4.extend(np.array(current_result["gt_labels"])[select_idx])
                bboxes4.extend(gt_bboxes[select_idx])

        img4 = cv2.resize(img4, (0, 0), fx=0.5, fy=0.5)
        bboxes4 = np.array(bboxes4, dtype=np.float32) / 2

        results["img"] = img4
        results["gt_bboxes"] = bboxes4.reshape([-1, 4])
        results["gt_labels"] = np.array(labels4, dtype=np.int64)

        return results

    def __call__(self, results):

        old_results = copy.deepcopy(results)

        if (
            np.random.random() <= self.prob
            and len(self.results_buffer) >= self.buffer_size
        ):
            results = self.mosaic_main(results)

        # update buffer
        if (
            "gt_bboxes" in old_results.keys() and len(old_results["gt_bboxes"])
        ) > 0 or (not self.no_empty_buffer):
            self.results_buffer.append(old_results)
            if len(self.results_buffer) > self.buffer_size:
                self.results_buffer.pop(0)

        return results


# MixUp code(mmdet/datasets/pipelines/transforms.py)
@COMMON_PIPELINES.register_module()
class MixUp(object):
    def __init__(self, prob=0.3, lambd=0.5):
        self.lambd = lambd
        self.prob = prob
        self.img2 = None
        self.boxes2 = None
        self.labels2 = None

    def __call__(self, results):
        img1, boxes1, labels1 = [results[k] for k in ("img", "gt_bboxes", "gt_labels")]

        if (
            np.random.random() < self.prob
            and self.img2 is not None
            and img1.shape[1] == self.img2.shape[1]
        ):

            height = max(img1.shape[0], self.img2.shape[0])
            width = max(img1.shape[1], self.img2.shape[1])
            mixup_image = np.zeros([height, width, 3], dtype="float32")
            mixup_image[: img1.shape[0], : img1.shape[1], :] = (
                img1.astype("float32") * self.lambd
            )
            mixup_image[
                : self.img2.shape[0], : self.img2.shape[1], :
            ] += self.img2.astype("float32") * (1.0 - self.lambd)
            mixup_image = mixup_image.astype("uint8")

            mixup_boxes = np.vstack((boxes1, self.boxes2))
            mixup_label = np.hstack((labels1, self.labels2))
            results["img"] = mixup_image
            results["gt_bboxes"] = mixup_boxes
            results["gt_labels"] = mixup_label
        else:
            pass

        self.img2 = img1
        self.boxes2 = boxes1
        self.labels2 = labels1
        return results


# BBoxJitter code, same place with above.
@COMMON_PIPELINES.register_module()
class BBoxJitter(object):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """

    def __init__(self, min_scale=0, max_scale=2):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.count = 0
        # ic("USE BBOX_JITTER")
        # ic(min, max)

    def bbox_jitter(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes
        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.0
            h = h * scale / 2.0
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0, img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0, img_shape[0] - 1)
        return jitter_bboxes

    def __call__(self, results):
        for key in results.get("bbox_fields", []):
            results[key] = self.bbox_jitter(results[key], results["img_shape"])
        return results

    def __repr__(self):
        return self.__class__.__name__ + "(bbox_jitter={}-{})".format(
            self.min_scale, self.max_scale
        )


@COMMON_PIPELINES.register_module()
class CopyAndPaste(object):
    def __init__(
        self,
        prob=0.5,
        max_put_num=3,
        paste_cats=[],
        border_size=100,
        buffer_size=20,
        max_occlusion_thresh=0.1,
        other_json="",
        min_box_len=5,
        random_crop_around_box_range=(500, 800),
        random_horizontal_flip=0.5,
        resize_range=(0.5, 1.5),
        smoothen_edge=True,
        use_blur=False,
    ):
        """[summary]

        Args:
            prob (float, optional): [description]. Defaults to 0.5.
            max_put_num (int, optional): [description]. Defaults to 3.
            paste_cats (list, optional): if [] use all cats, else check whether cat in cats
                cat are after cat_to_labels. Defaults to [].
            border_size (int, optional): [description]. Defaults to 100.
            buffer_size (int, optional): [description]. Defaults to 20.
            max_occlusion_thresh (float, optional): [description]. Defaults to 0.1.
            other_json (str, optional): [description]. Defaults to ''.
            min_box_len (int, optional): [description]. Defaults to 20.
            random_crop_around_box_range (tuple, optional): [description]. Defaults to (500, 800).
            random_horizontal_flip (float, optional): [description]. Defaults to 0.5.
            resize_range (tuple, optional): [description]. Defaults to (0.5, 1.5).
            smoothen_edge (bool, optional): [description]. Defaults to True.
            use_blur (bool, optional): [description]. Defaults to False.
        """

        # params
        self.prob = prob
        self.max_put_num = max_put_num
        self.paste_cats = paste_cats
        self.border_size = border_size
        self.buffer_size = buffer_size
        self.max_occlusion_thresh = max_occlusion_thresh
        self.other_json = other_json
        self.min_box_len = min_box_len
        self.random_crop_around_box_range = random_crop_around_box_range
        self.random_horizontal_flip = random_horizontal_flip
        self.resize_range = resize_range
        self.smoothen_edge = smoothen_edge
        self.use_blur = use_blur

        # buffer
        self.buffers = []
        self.other_buffers = []

    def cut_image(self, img, bbox, shape):
        """cut image if bbox is target image
        Args:
            img (np.array): patch
            bbox (list): [xmin, ymin, xmax, ymax]
            shape (list]): [height, width]
        Returns:
            cropped image
        """
        if bbox[0] < 0:
            img = img[:, abs(bbox[0]) :, :]
        elif bbox[2] > shape[1]:
            img = img[:, : shape[1] - bbox[2], :]

        if bbox[1] < 0:
            img = img[abs(bbox[1]) :, :, :]
        elif bbox[3] > shape[0]:
            img = img[: shape[0] - abs(bbox[1]), :, :]
        return img

    def cut_box(self, bbox, shape):
        """cut box if box is out of shape"""
        bbox[0] = min(max(0, bbox[0]), shape[1])
        bbox[1] = min(max(0, bbox[1]), shape[0])
        bbox[2] = min(max(0, bbox[2]), shape[1])
        bbox[3] = min(max(0, bbox[3]), shape[0])
        return bbox

    def _del_small_area_bboxes(self, boxes, box_len):
        """Delete small area boxes

        Args:
            boxes (np.array): np.array with shape nx4 for mat [xmin, ymin, xmax, ymax]
        """
        width = (boxes[:, 2] - boxes[:, 0]) > box_len
        height = (boxes[:, 3] - boxes[:, 1]) > box_len
        select_indices = width & height
        return boxes[select_indices], select_indices

    def random_crop_around_box(
        self, image, target_box, other_bboxes, random_crop_around_box_range, **kwargs
    ):
        """random crop around box.
        Args:
            image (np.array): [hxwxc]
            target_box: main boxes to crop
            boxes (list): get nx4 xmin, ymin, xmax, ymax
            random_crop_around_box_range (tuple): random around crop range
        Returns:
            cropped image and box coco[xmin, ymin, xmax, ymax]
            target_box after crop
            other_bboxes after crop
            select indices of other boxes
        """
        xmin, ymin, xmax, ymax = target_box
        crop_dis_l = np.random.randint(
            random_crop_around_box_range[0], random_crop_around_box_range[1]
        )
        crop_dis_r = np.random.randint(
            random_crop_around_box_range[0], random_crop_around_box_range[1]
        )
        crop_dis_t = np.random.randint(
            random_crop_around_box_range[0], random_crop_around_box_range[1]
        )
        crop_dis_d = np.random.randint(
            random_crop_around_box_range[0], random_crop_around_box_range[1]
        )

        new_xmin = max(0, xmin - crop_dis_l)
        new_ymin = max(0, ymin - crop_dis_t)
        new_xmax = min(image.shape[1], xmax + crop_dis_r)
        new_ymax = min(image.shape[0], ymax + crop_dis_d)

        dst = image[new_ymin:new_ymax, new_xmin:new_xmax]
        other_bboxes[:, 0] = np.maximum(other_bboxes[:, 0], new_xmin)
        other_bboxes[:, 1] = np.maximum(other_bboxes[:, 1], new_ymin)
        other_bboxes[:, 2] = np.minimum(other_bboxes[:, 2], new_xmax)
        other_bboxes[:, 3] = np.minimum(other_bboxes[:, 3], new_ymax)

        other_bboxes[:, 0] -= new_xmin
        other_bboxes[:, 1] -= new_ymin
        other_bboxes[:, 2] -= new_xmin
        other_bboxes[:, 3] -= new_ymin
        other_bboxes, other_indices = self._del_small_area_bboxes(
            other_bboxes, self.min_box_len
        )

        target_box = np.array(
            [
                xmin - new_xmin,
                ymin - new_ymin,
                xmax - new_xmin,
                ymax - new_ymin,
            ]
        )

        return dst, target_box, other_bboxes, other_indices

    def blur(self, img, kernel_size, is_vertical=False):
        if kernel_size == 0:
            return img
        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_h = np.copy(kernel_v)
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        # Normalize
        kernel_v /= kernel_size
        kernel_h /= kernel_size
        if is_vertical:
            img = cv2.filter2D(img, -1, kernel_v)
        else:
            img = cv2.filter2D(img, -1, kernel_h)
        return img

    def box_intersection(self, boxA, boxB):
        """Calculate two box intersection.
        Args:
            boxA (list): [xmin, ymin, xmax, ymax]
            boxB (list): [xmin, ymin, xmax, ymax]
        Returns:
            intersection or 0 if no intersection
        """
        # 相交矩形
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea

    def box_area(self, box):
        """box area.
        Args:
            box (list): [xmin, ymin, xmax, ymax]
        Returns:
            box area
        """

        return (box[2] - box[0]) * (box[3] - box[1])

    def occlusion(self, boxA, boxB):
        interArea = self.box_intersection(boxA, boxB)

        boxAArea = self.box_area(boxA)
        boxBArea = self.box_area(boxB)
        iou = interArea / float(min(boxAArea, boxBArea))
        return iou

    def max_occlusion(self, target_box, boxes):
        """Calculate max occlusion between target box and boxes.
        Args:
            target_box (np.array): [xmin, ymin, xmax, ymax]
            boxes (np.array): list of box
        Returns:
            max_occlusion
        """
        ret = 0
        for box in boxes:
            iou = self.occlusion(box, target_box)
            ret = max(ret, iou)
        return ret

    def create_rectrangle_gaussian_masks(
        self, base_shape=[50, 50, 3], border=5, kernel=(21, 21), sigma_x=0
    ):
        """Create rectangle mask with gradient border.
        Args:
            base_shape (list, optional): mask shape. Defaults to [50, 50, 3].
            border (int, optional): border. Defaults to 5.
            kernel (tuple, optional): gaussian kernel. Defaults to (21, 21).
            sigma_x (int, optional): signmax for gaussian blur. Defaults to 0.
        Returns:
            mask with shape as base_shape
        """
        img = np.zeros(base_shape, dtype=np.float32)
        img[border : base_shape[0] - border, border : base_shape[1] - border, :] = 1
        return cv2.GaussianBlur(img, kernel, sigma_x)

    def copy_and_save(self, results):
        """copy bbox and save to buffer

        Args:
            results (dict): input image results

        Internal Args:
            paste_cats(list): if empty use all cats


        """

        if "gt_bboxes" not in results.keys() or not len(results["gt_bboxes"]):
            return

        # read data, gt_boxes and labels numpy array
        gt_bboxes = results["gt_bboxes"]
        gt_labels = results["gt_labels"]
        source_image = results["img"]

        for idx, xyxy in enumerate(gt_bboxes):
            if len(self.paste_cats) and gt_labels[idx] not in self.paste_cats:
                continue

            # select target box
            xmin, ymin, xmax, ymax = (
                int(xyxy[0]),
                int(xyxy[1]),
                int(xyxy[2]),
                int(xyxy[3]),
            )
            label = gt_labels[idx]

            # collect other boxes
            other_bboxes = np.delete(gt_bboxes, idx, 0)
            other_labels = np.delete(gt_labels, idx, 0)

            # random crop around target box
            dst, target_box, other_bboxes, select_idx = self.random_crop_around_box(
                source_image,
                target_box=[xmin, ymin, xmax, ymax],
                other_bboxes=other_bboxes,
                random_crop_around_box_range=self.random_crop_around_box_range,
            )

            bboxes = np.append(target_box.reshape([-1, 4]), other_bboxes, axis=0)
            labels = np.append(label, other_labels[select_idx])

            if not len(bboxes):
                continue

            #### add transforms
            # random horizontal flip
            if np.random.random() < self.random_horizontal_flip:
                dst = dst[:, ::-1, :]
                xmax = dst.shape[1] - bboxes[:, 0]
                xmin = dst.shape[1] - bboxes[:, 2]
                bboxes[:, 0] = xmin
                bboxes[:, 2] = xmax

            # random resize
            resize_ratio = (
                self.resize_range[1] - self.resize_range[0]
            ) * np.random.random() + self.resize_range[0]
            dst = cv2.resize(dst, (0, 0), fx=resize_ratio, fy=resize_ratio)
            bboxes = bboxes * resize_ratio

            self.buffers.append((dst, bboxes, labels))

    def select_and_paste(self, results):
        target_image = copy.deepcopy(results["img"])
        gt_bboxes = copy.deepcopy(results["gt_bboxes"])
        gt_occlu_boxes = copy.deepcopy(results["gt_bboxes"])
        gt_labels = copy.deepcopy(results["gt_labels"])

        for _ in range(self.max_put_num):
            select_idx = np.random.choice(range(len(self.buffers)))
            select_img, buffer_bboxes, buffer_labels = self.buffers[select_idx]
            select_bboxes = copy.deepcopy(buffer_bboxes)
            select_labels = copy.deepcopy(buffer_labels)

            # add blur
            if self.use_blur:
                blur_type = np.random.random()
                if blur_type < 0.3:
                    blur_value = np.random.randint(1, 8)
                    dst = cv2.blur(dst, (blur_value, blur_value))
                elif blur_type < 0.6:
                    blur_value = np.random.choice([3, 5, 7])
                    dst = cv2.GaussianBlur(dst, (blur_value, blur_value), 2)
                elif blur_type < 0.7:
                    dst = self.blur(dst, np.random.randint(1, 8))

            # random put
            w_put = np.random.randint(0, target_image.shape[1])
            h_put = np.random.randint(0, target_image.shape[0])
            width = np.max(select_bboxes[:, 2]) - np.min(select_bboxes[:, 0])
            height = np.max(select_bboxes[:, 3]) - np.min(select_bboxes[:, 1])

            # select put position
            put_pos_xmin = 0
            put_pos_ymin = 0
            put_pos_xmax = 0
            put_pos_ymax = 0
            if target_image.shape[1] - w_put < width:
                put_pos_xmax = w_put
                put_pos_xmin = w_put - select_img.shape[1]
            else:
                put_pos_xmin = w_put
                put_pos_xmax = w_put + select_img.shape[1]

            if target_image.shape[0] - h_put < height:
                put_pos_ymax = h_put
                put_pos_ymin = h_put - select_img.shape[0]
            else:
                put_pos_ymin = h_put
                put_pos_ymax = h_put + select_img.shape[0]

            # crop box and crop image
            select_img = self.cut_image(
                select_img,
                [put_pos_xmin, put_pos_ymin, put_pos_xmax, put_pos_ymax],
                target_image.shape[:2],
            )
            select_bboxes[:, 0] += put_pos_xmin
            select_bboxes[:, 1] += put_pos_ymin
            select_bboxes[:, 2] += put_pos_xmin
            select_bboxes[:, 3] += put_pos_ymin
            put_pos_xmin, put_pos_ymin, put_pos_xmax, put_pos_ymax = self.cut_box(
                [put_pos_xmin, put_pos_ymin, put_pos_xmax, put_pos_ymax],
                target_image.shape[:2],
            )

            # for industry, not allowed to past out side of the image
            if (
                min(select_bboxes[:, 0]) < self.border_size
                or min(select_bboxes[:, 1]) < self.border_size
                or max(select_bboxes[:, 2]) > target_image.shape[1] - self.border_size
                or max(select_bboxes[:, 3]) > target_image.shape[0] - self.border_size
            ):
                continue

            # check occlusion
            if (
                self.max_occlusion(
                    [put_pos_xmin, put_pos_ymin, put_pos_xmax, put_pos_ymax],
                    gt_occlu_boxes,
                )
                > self.max_occlusion_thresh
            ):
                continue

            # update and next loop
            gt_occlu_boxes = np.append(
                gt_occlu_boxes,
                np.array([[put_pos_xmin, put_pos_ymin, put_pos_xmax, put_pos_ymax]]),
                axis=0,
            )
            gt_bboxes = np.append(gt_bboxes, select_bboxes, axis=0)
            gt_labels = np.append(gt_labels, select_labels)

            # smoothen edge
            if self.smoothen_edge:
                mask = self.create_rectrangle_gaussian_masks(
                    base_shape=select_img.shape
                )
                target_image[put_pos_ymin:put_pos_ymax, put_pos_xmin:put_pos_xmax] = (
                    target_image[put_pos_ymin:put_pos_ymax, put_pos_xmin:put_pos_xmax]
                    * (1 - mask)
                    + mask * select_img
                )
            else:
                target_image[
                    put_pos_ymin:put_pos_ymax, put_pos_xmin:put_pos_xmax
                ] = select_img

        results["img"] = target_image
        results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32)
        results["gt_labels"] = np.array(gt_labels, dtype=np.int64)
        return results

    def __call__(self, results):
        new_results = copy.deepcopy(results)

        if random.random() <= self.prob and len(self.buffers) >= self.buffer_size:
            new_results = self.select_and_paste(new_results)

        # update buffer
        if "gt_bboxes" in results.keys() and len(results["gt_bboxes"]):
            self.copy_and_save(results)
            for idx in range(len(self.buffers) - self.buffer_size):
                self.buffers.pop(0)

        return new_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"(buffer_size={self.buffer_size}, "
        repr_str += f"(other_json={self.other_json}, "
        return repr_str


@COMMON_PIPELINES.register_module()
class CopyAndPasteV2(object):
    def __init__(
        self,
        cats="all",
        paste_prob=0.5,
        crop_prob=1.0,
        random_crop_extend_x=(50, 300),
        random_crop_extend_y=(50, 300),
        resize_range=(0.5, 1.5),
        pos_ratio=0.9,
        neg_ratio=0.3,
        buffer_size=20,
        max_paste_number=3,
        crop_try_number=10,
        paste_try_number=10,
        use_blur=False,
    ):
        # params
        self.paste_prob = paste_prob
        self.crop_prob = crop_prob
        self.max_paste_number = max_paste_number
        self.crop_try_number = crop_try_number
        self.paste_try_number = paste_try_number
        self.cats = cats
        self.buffer_size = buffer_size

        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio

        self.random_crop_extend_x = random_crop_extend_x
        self.random_crop_extend_y = random_crop_extend_y

        self.resize_range = resize_range

        self.use_blur = use_blur

        # buffer
        self.buffers = []

    def get_random_crop_bbox(self, image, target_box):
        h, w, _ = image.shape
        xmin, ymin, xmax, ymax = target_box
        crop_xmin = max(0, xmin - np.random.randint(*self.random_crop_extend_x))
        crop_xmax = min(w, xmax + np.random.randint(*self.random_crop_extend_x))
        crop_ymin = max(0, ymin - np.random.randint(*self.random_crop_extend_y))
        crop_ymax = min(h, ymax + np.random.randint(*self.random_crop_extend_y))
        return np.array([crop_xmin, crop_ymin, crop_xmax, crop_ymax])

    def get_occlusion_bbox(self, crop_bbox, gt_bboxes, pos_ratio, neg_ratio):
        valid_flags = []
        for gt_bbox in gt_bboxes:
            occlusion = float(
                self.box_intersection(gt_bbox, crop_bbox)
            ) / self.box_area(gt_bbox)
            if occlusion <= neg_ratio:
                valid_flags.append(0)
            elif occlusion >= pos_ratio:
                valid_flags.append(1)
            else:
                # occlusion > neg_ratio and occlusion < pos_ratio:
                valid_flags.append(-1)
        return np.array(valid_flags)

    def blur(self, img, kernel_size, is_vertical=False):
        if kernel_size == 0:
            return img
        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_h = np.copy(kernel_v)
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        # Normalize
        kernel_v /= kernel_size
        kernel_h /= kernel_size
        if is_vertical:
            img = cv2.filter2D(img, -1, kernel_v)
        else:
            img = cv2.filter2D(img, -1, kernel_h)
        return img

    def box_intersection(self, boxa, boxb):
        # ç›¸äº¤çŸ©å½¢
        xa = max(boxa[0], boxb[0])
        ya = max(boxa[1], boxb[1])
        xb = min(boxa[2], boxb[2])
        yb = min(boxa[3], boxb[3])
        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        return inter_area

    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def create_rectrangle_gaussian_masks(
        self, base_shape=(50, 50, 3), border=5, kernel=(21, 21), sigma_x=0
    ):
        img = np.zeros(base_shape, dtype=np.float32)
        img[border : base_shape[0] - border, border : base_shape[1] - border, :] = 1
        return cv2.GaussianBlur(img, kernel, sigma_x)

    def copy_and_save(self, results):

        if "gt_bboxes" not in results.keys() or not len(results["gt_bboxes"]) > 0:
            return

        gt_bboxes = results["gt_bboxes"]
        gt_labels = results["gt_labels"]
        image = results["img"]

        for idx, target_box in enumerate(gt_bboxes):
            if (
                self.cats == "all"
                or gt_labels[idx] in self.cats
                and np.random.random() <= self.crop_prob
            ):

                for _ in range(self.crop_try_number):
                    crop_bbox = self.get_random_crop_bbox(image, target_box)
                    valid_flags = self.get_occlusion_bbox(
                        crop_bbox, gt_bboxes, self.pos_ratio, self.neg_ratio
                    )
                    if -1 not in valid_flags:
                        break
                if -1 in valid_flags:
                    continue

                crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox

                dst = image[int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)]

                save_bboxes = gt_bboxes - np.array([crop_x1, crop_y1, crop_x1, crop_y1])
                save_bboxes[:, 0] = np.clip(save_bboxes[:, 0], 0, None)
                save_bboxes[:, 1] = np.clip(save_bboxes[:, 1], 0, None)
                save_bboxes[:, 2] = np.clip(save_bboxes[:, 2], None, crop_x2 - crop_x1)
                save_bboxes[:, 3] = np.clip(save_bboxes[:, 3], None, crop_y2 - crop_y1)

                save_bboxes = save_bboxes[valid_flags == 1]
                save_gt_labels = gt_labels[valid_flags == 1]

                if len(save_gt_labels) > 0:
                    self.buffers.append((dst, save_bboxes, save_gt_labels))

    def paste_augmentation(self, img, bboxes):
        if self.use_blur:
            blur_type = np.random.random()
            if blur_type < 0.3:
                blur_value = np.random.randint(1, 8)
                img = cv2.blur(img, (blur_value, blur_value))
            elif blur_type < 0.6:
                blur_value = np.random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (blur_value, blur_value), 2)
            elif blur_type < 0.7:
                img = self.blur(img, np.random.randint(1, 8))

        # random resize
        resize_ratio = np.random.uniform(*self.resize_range)
        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        bboxes = np.round(bboxes * resize_ratio)

        return img, bboxes

    def select_and_paste(self, results):
        origin_image = copy.deepcopy(results["img"])
        gt_bboxes = copy.deepcopy(results["gt_bboxes"])
        gt_labels = copy.deepcopy(results["gt_labels"])

        for _ in range(self.paste_try_number):
            if len(self.buffers) == 0:
                break
            select_idx = np.random.choice(range(len(self.buffers)))
            select_img, select_bboxes, select_labels = self.buffers[select_idx]
            select_img, select_bboxes = self.paste_augmentation(
                select_img, select_bboxes
            )
            select_h, select_w, _ = select_img.shape
            origin_h, origin_w, _ = origin_image.shape
            if origin_w - select_w <= 0 or origin_h - select_h <= 0:
                continue
            x1_put = np.random.randint(0, origin_w - select_w)
            y1_put = np.random.randint(0, origin_h - select_h)
            x2_put = x1_put + select_w
            y2_put = y1_put + select_h
            crop_bbox = [x1_put, y1_put, x2_put, y2_put]
            valid_flags = self.get_occlusion_bbox(
                crop_bbox, gt_bboxes, 1 - self.neg_ratio, 1 - self.pos_ratio
            )
            if -1 not in valid_flags:
                mask = self.create_rectrangle_gaussian_masks(
                    base_shape=select_img.shape
                )
                origin_image[y1_put:y2_put, x1_put:x2_put] = (
                    origin_image[y1_put:y2_put, x1_put:x2_put] * (1 - mask)
                    + mask * select_img
                )
                gt_bboxes = gt_bboxes[valid_flags == 0]
                gt_labels = gt_labels[valid_flags == 0]
                select_bboxes += [x1_put, y1_put, x1_put, y1_put]
                gt_bboxes = np.concatenate([gt_bboxes, select_bboxes])
                gt_labels = np.concatenate([gt_labels, select_labels])
                self.buffers.pop(select_idx)
                break

        results["img"] = origin_image
        results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape([-1, 4])
        results["gt_labels"] = np.array(gt_labels, dtype=np.int64)
        return results

    def __call__(self, results):
        new_results = copy.deepcopy(results)

        if (
            np.random.random() <= self.crop_prob
            and len(self.buffers) >= self.buffer_size
        ):
            paste_number = np.random.randint(1, self.max_paste_number)
            for _ in range(paste_number):
                new_results = self.select_and_paste(new_results)

        self.copy_and_save(results)
        for _ in range(len(self.buffers) - self.buffer_size):
            self.buffers.pop(0)

        return new_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(buffer_size={self.buffer_size}, "
        return repr_str



@COMMON_PIPELINES.register_module()
class RotateAlign:
    """ Rotate image to all height > width
    """

    def _rotate_bbox(self, bboxes, image_shape):
        rotated = bboxes.copy()
        rotated[:, 0] = image_shape[0] - bboxes[:, 3]
        rotated[:, 1] = bboxes[:, 0]
        rotated[:, 2] = image_shape[0] - bboxes[:, 1]
        rotated[:, 3] = bboxes[:, 2]
        return rotated

    def __call__(self, results) -> None:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: rotated image
        """
        rotated = False
        image_shape = None
        for key in results.get('img_fields', ['img']):
            image = results[key]
            if image.shape[1] > image.shape[0]:
                image_shape = image.shape
                results[key] = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
                results['img_shape'] = results[key].shape
                rotated = True

        if rotated:
            for key in results.get('bbox_fields', []):
                results[key] = self._rotate_bbox(results[key], image_shape)

        return results


@COMMON_PIPELINES.register_module()
class MSResize(Resize):
    """Resize changed for mstrain, when image scale is changing
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            ### This has changed, ratio_range according to incomming image scale
            # mode 1: given a scale and a range of image ratio
            # assert len(self.img_scale) == 1
            pass
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    def random_sample_ratio(self, img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        # scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return ratio, None


@COMMON_PIPELINES.register_module()
class RandomCropAroundBbox(RandomCrop):
    def __init__(self, min_box_len=2, keep_ratio=False, **kwargs):
        # min_box_len is the min box len after crop
        self.min_box_len = min_box_len
        self.keep_ratio = keep_ratio
        super().__init__(**kwargs)

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            # if crop size > image shape, use image shape
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = ((bboxes[:, 2] - bboxes[:, 0]) > self.min_box_len) & (
                (bboxes[:, 3] - bboxes[:, 1]) > self.min_box_len)

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            ## add len(bboxes) because blank image doesn't need crop
            if (key == 'gt_bboxes' and len(bboxes) and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
                # NOTE. updated by ryanwfu, support attr crop
                if 'gt_attrs' in results: results['gt_attrs'] = np.array(results['gt_attrs'])[valid_inds].tolist()

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _crop_data_around_bbox(
            self, results, crop_size):
        """random crop around box.
        Args:
            image (np.array): [hxwxc]
            box (list): get ymin, xmin, ymax, xmax and use as xmin, ymin, xmax, ymax
            random_crop_around_box_range (tuple): random around crop range
        Returns:
            cropped image and box coco[xmin, ymin, xmax, ymax]
        """
        assert crop_size[0] > 0 and crop_size[1] > 0

        # 1 select crop bbox
        bbox = results['gt_bboxes'][np.random.randint(len(results['gt_bboxes']))]

        # 2 crop size around bbox
        img_shape = results[results.get('img_fields', ['img'])[0]].shape
        xmin, ymin, xmax, ymax = bbox
        crop_around_size = (
            max(1, crop_size[0] - (ymax - ymin)),
            max(1, crop_size[1] - (xmax - xmin)))
        crop_y1 = max(0, ymin - np.random.randint(crop_around_size[0]))
        if crop_y1 + crop_size[0] > img_shape[0]:
            crop_y1 = max(0, img_shape[0] - crop_size[0])
        crop_y2 = min(img_shape[0], crop_y1 + crop_size[0])
        crop_x1 = max(0, xmin - np.random.randint(crop_around_size[1]))
        if crop_x1 + crop_size[1] > img_shape[1]:
            crop_x1 = max(0, img_shape[1] - crop_size[1])
        crop_x2 = min(img_shape[1], crop_x1 + crop_size[1])
        crop_y1 = int(crop_y1)
        crop_y2 = int(crop_y2)
        crop_x1 = int(crop_x1)
        crop_x2 = int(crop_x2)

        # 3 crop  image
        for key in results.get('img_fields', ['img']):
            img = results[key]
            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([crop_x1, crop_y1, crop_x1, crop_y1],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = ((bboxes[:, 2] - bboxes[:, 0]) > self.min_box_len) & (
                (bboxes[:, 3] - bboxes[:, 1]) > self.min_box_len)

            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
                # NOTE. updated by ryanwfu, support attr crop
                if 'gt_attrs' in results: results['gt_attrs'] = np.array(results['gt_attrs'])[valid_inds].tolist()

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if "gt_bboxes" not in results.keys() or not len(results['gt_bboxes']):
            return super().__call__(results)
        image_size = results['img'].shape[:2]

        # NOTE. add keep ratio pattern @choasliu
        if self.keep_ratio:
            ratio = np.sqrt((self.crop_size[0] * self.crop_size[1]) / (image_size[0] * image_size[1]))
            self.crop_size = (int(ratio * image_size[0]), int(ratio * image_size[1]))

        crop_size = self._get_crop_size(image_size)

        # NOTE. the old crop method may return None. @lloydwu
        results = self._crop_data_around_bbox(results, crop_size)
        return results


@COMMON_PIPELINES.register_module()
class CropAlongLongerSideFromDataset:

    def __init__(
            self,
            crop_jitter=0,
            min_box_len=2,
            bbox_clip_border=True,
            recompute_bbox=False,
            allow_negative_crop=False):
        """ check random crop for init comments

        crop_jitter(int): crop jitter, crop bbox  +- crop_jitter
        """
        self.crop_jitter = crop_jitter
        self.min_box_len = min_box_len
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        self.allow_negative_crop = allow_negative_crop

        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _get_crop_jitter(self, crop_size, min_val=0, max_val=1e6):
        """ get crop jitter

        Args:
            crop_size (tuple): crop size to be jittered

        Returns:
            crop_size: jittered crop_size +- self.crop_jitter, each element
                of crop size is >= 0
        """
        if self.crop_jitter > 0:
            jitter_value = int((np.random.random() * 2 - 1) * self.crop_jitter)
            if jitter_value > 0:
                jitter_diff = min([max_val - x for x in crop_size])
                jitter_value = min(jitter_diff, jitter_value)
            else:
                jitter_diff = max([min_val - x for x in crop_size])
                jitter_value = max(jitter_diff, jitter_value)
            return tuple([max(x + jitter_value, 0) for x in crop_size])
        return crop_size

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] >= 0 and crop_size[1] >= 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            crop_x1 = 0
            crop_y1 = 0
            crop_x2 = img.shape[1]
            crop_y2 = img.shape[0]
            if img.shape[0] >= img.shape[1]:
                crop_size = self._get_crop_jitter(
                    crop_size, max_val=img.shape[0])
                crop_y1, crop_y2 = crop_size
            else:
                crop_size = self._get_crop_jitter(
                    crop_size, max_val=img.shape[1])
                crop_x1, crop_x2 = crop_size
            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([crop_x1, crop_y1, crop_x1, crop_y1],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = ((bboxes[:, 2] - bboxes[:, 0]) > self.min_box_len) & (
                (bboxes[:, 3] - bboxes[:, 1]) > self.min_box_len)
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and len(bboxes) and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
                # NOTE. updated by ryanwfu, support attr crop
                if 'gt_attrs' in results: results['gt_attrs'] = np.array(results['gt_attrs'])[valid_inds].tolist()

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, results):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_meta
            xxxs

        Returns:
            crop_size (tuple): (crop_start, crop_end) in absolute pixels.
                the crop will be long longer side
        """
        return results['img_info'].get("crop_range", None)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        crop_size = self._get_crop_size(results)
        if crop_size is None:
            return results
        results = self._crop_data(
            results, crop_size, allow_negative_crop=self.allow_negative_crop)
        return results



@COMMON_PIPELINES.register_module()
class CropRectangleFromDataset(CropAlongLongerSideFromDataset):

    def _get_crop_jitter(self, crop_size, min_val=0, max_val=1e6):
        """ get crop jitter

        Args:
            crop_size (tuple): crop size to be jittered
            min_val/max_val (int): upper and lower boundary for crop usually
                is  0 and image.shape

        Returns:
            crop_size: jittered crop_size +- self.crop_jitter, each element
                of crop size is >= 0
        """
        if self.crop_jitter > 0:
            jitter_value = int((np.random.random() * 2 - 1) * self.crop_jitter)
            if jitter_value > 0:
                jitter_diff = min([max_val - x for x in crop_size])
                jitter_value = min(jitter_diff, jitter_value)
            else:
                jitter_diff = max([min_val - x for x in crop_size])
                jitter_value = max(jitter_diff, jitter_value)
            return tuple([max(x + jitter_value, 0) for x in crop_size])
        return crop_size

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute position after cropping, (h, w).
                or four value min and max
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] >= 0 and crop_size[1] >= 0
        assert (
            len(crop_size) == 4
        ), "crop_range from dataset must have length four [xmin, ymin, xmax, ymax] got {} instead".format(
            crop_size)

        for key in results.get('img_fields', ['img']):
            img = results[key]
            crop_x1 = 0
            crop_y1 = 0
            crop_x2 = img.shape[1]
            crop_y2 = img.shape[0]
            crop_y1, crop_y2 = self._get_crop_jitter(
                (crop_size[1], crop_size[3]), max_val=img.shape[0])
            crop_x1, crop_x2 = self._get_crop_jitter(
                (crop_size[0], crop_size[2]), max_val=img.shape[1])

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([crop_x1, crop_y1, crop_x1, crop_y1],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = ((bboxes[:, 2] - bboxes[:, 0]) > self.min_box_len) & (
                (bboxes[:, 3] - bboxes[:, 1]) > self.min_box_len)
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and len(bboxes) and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
                # NOTE. updated by ryanwfu, support attr crop
                if 'gt_attrs' in results: results['gt_attrs'] = np.array(results['gt_attrs'])[valid_inds].tolist()

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results
