import copy
import inspect
import math
import time
import warnings

import cv2
import mmcv
import numpy as np
from .transforms import COMMON_PIPELINES

import cv2
import torch
import numbers

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from threading import Lock
mean_gpu_mat_map = dict()
stdinv_gpu_mat_map = dict()
map_update_lock = Lock()


@COMMON_PIPELINES.register_module()
class Pad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None
        print(f"Enable gpu for Pad")

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                if isinstance(results[key], cv2.cuda_GpuMat) or isinstance(results[key], torch.Tensor):
                    pad_h = int(np.ceil(results[key].size()[1] / self.size_divisor)) * self.size_divisor
                    pad_w = int(np.ceil(results[key].size()[0] / self.size_divisor)) * self.size_divisor
                    if pad_h == results[key].size()[1] and pad_w == results[key].size()[0]:
                        padded_img = results[key]
                    else:
                        if isinstance(results[key], torch.Tensor) and results[key].shape[-1] != 3:  # Multi Light
                            padded_img = mmcv.impad_to_multiple(
                                results[key].cpu().numpy(), self.size_divisor, pad_val=pad_val)
                        else:
                            padded_img = self._impad(
                                results[key], shape=(pad_h, pad_w), pad_val=pad_val)
                else:
                    padded_img = mmcv.impad_to_multiple(
                        results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        if isinstance(padded_img, cv2.cuda_GpuMat):
            results['pad_shape'] = (padded_img.size()[1],) + (padded_img.size()[0],) + (3,)
        else:
            results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

    def _impad(self,
               img,
               *,
               shape=None,
               padding=None,
               pad_val=0,
               padding_mode='constant'):
        """Pad the given image to a certain shape or pad on all sides with
        specified padding mode and padding value.

        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w). Default: None.
            padding (int or tuple[int]): Padding on each border. If a single int is
                provided this is used to pad all borders. If tuple of length 2 is
                provided this is the padding on left/right and top/bottom
                respectively. If a tuple of length 4 is provided this is the
                padding for the left, top, right and bottom borders respectively.
                Default: None. Note that `shape` and `padding` can not be both
                set.
            pad_val (Number | Sequence[Number]): Values to be filled in padding
                areas when padding_mode is 'constant'. Default: 0.
            padding_mode (str): Type of padding. Should be: constant, edge,
                reflect or symmetric. Default: constant.

                - constant: pads with a constant value, this value is specified
                    with pad_val.
                - edge: pads with the last value at the edge of the image.
                - reflect: pads with reflection of image without repeating the
                    last value on the edge. For example, padding [1, 2, 3, 4]
                    with 2 elements on both sides in reflect mode will result
                    in [3, 2, 1, 2, 3, 4, 3, 2].
                - symmetric: pads with reflection of image repeating the last
                    value on the edge. For example, padding [1, 2, 3, 4] with
                    2 elements on both sides in symmetric mode will result in
                    [2, 1, 1, 2, 3, 4, 4, 3]

        Returns:
            ndarray: The padded image.
        """

        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            padding = (0, 0, shape[1] - img.size()[0], shape[0] - img.size()[1])

        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == 1
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                            f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            img = cv2.cuda_GpuMat(img)
            img = img.convertTo(cv2.CV_32FC3, img)

        img = cv2.cuda.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)

        return img

@COMMON_PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
        enable_gpu (bool): Whether normalize with gpu.
        output_numpy (bool): Whether output numpy image on cpu.
    """

    def _generate_mean_std(self, img_size):
        mean_val = self.mean.reshape(1, 1, -1)
        mean_repeat_val = np.repeat(np.repeat(mean_val, img_size[1], axis=1),
                                    img_size[0], axis=0)

        stdinv_val = 1 / self.std.reshape(1, 1, -1)
        stdinv_repeat_val = np.repeat(np.repeat(stdinv_val, img_size[1], axis=1),
                                      img_size[0], axis=0)

        if self.mode == 'cv2.cuda':
            mean_gpu_mat = cv2.cuda_GpuMat(mean_repeat_val)
            stdinv_gpu_mat = cv2.cuda_GpuMat(stdinv_repeat_val)
        else:
            mean_gpu_mat = torch.tensor(mean_repeat_val).cuda()
            stdinv_gpu_mat = torch.tensor(stdinv_repeat_val).cuda()

        return mean_gpu_mat, stdinv_gpu_mat

    def _compute_md5(self, src_data, extra_info):
        import hashlib
        md5 = hashlib.md5()  # ignore
        md5.update(src_data.data.tobytes())
        md5.update(extra_info.encode("utf-8"))
        return md5.hexdigest()

    def __init__(self, mean, std, to_rgb=True, mode='cv2.cuda', output_numpy=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        assert mode in ['cpu', 'cv2.cuda', 'torch.cuda']
        self.enable_gpu = mode in ['cv2.cuda', 'torch.cuda']
        self.output_numpy = output_numpy
        self.mode = mode
        print(f"Use {self.mode} for Normalize")

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            if self.enable_gpu:
                if self.mode == 'cv2.cuda':
                    if results[key].dtype != np.uint8:
                        img = cv2.cuda_GpuMat(results[key].astype(np.uint8))
                    else:
                        img = cv2.cuda_GpuMat(results[key])
                    if self.to_rgb:
                        cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                    img = img.convertTo(cv2.CV_32FC3, img)
                    cur_img_size = img.size()[::-1]
                else:
                    if self.to_rgb:
                        results[key] = cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB)
                    if results[key].dtype != np.uint8:
                        img = torch.tensor(results[key].astype(np.uint8).astype(np.float32)).cuda()
                    else:
                        img = torch.tensor(results[key].astype(np.float32)).cuda()
                    cur_img_size = img.shape

                mean_key = self._compute_md5(self.mean, str(cur_img_size))
                std_key = self._compute_md5(self.std, str(cur_img_size))
                global mean_gpu_mat_map, stdinv_gpu_mat_map
                map_update_lock.acquire()
                if mean_key not in mean_gpu_mat_map or std_key not in stdinv_gpu_mat_map:
                    print(">>> realloc mean and std")
                    mean_gpu_mat, stdinv_gpu_mat = self._generate_mean_std(cur_img_size)
                    mean_gpu_mat_map[mean_key] = mean_gpu_mat
                    stdinv_gpu_mat_map[std_key] = stdinv_gpu_mat
                
                if self.mode == 'cv2.cuda':
                    cv2.cuda.subtract(img, mean_gpu_mat_map[mean_key], img)
                    cv2.cuda.multiply(img, stdinv_gpu_mat_map[std_key], img)
                else:
                    img = torch.subtract(img, mean_gpu_mat_map[mean_key])
                    img = torch.multiply(img, stdinv_gpu_mat_map[std_key])
                map_update_lock.release()

                if self.output_numpy:
                    results[key] = img.download() if self.mode == 'cv2.cuda' else img.cpu().numpy()
                else:
                    results[key] = img
            else:
                results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                                self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb}, mode={self.mode}, enable_gpu={self.enable_gpu}, output_numpy={self.output_numpy})'
        return repr_str
