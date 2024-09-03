from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from .transforms import COMMON_PIPELINES

import cv2
import time

class CudaArrayInterface:
    def __init__(self, gpu_mat):
        w, h = gpu_mat.size()
        c = gpu_mat.channels()
        type_map = {
            cv2.CV_8U: "u1", cv2.CV_8S: "i1",
            cv2.CV_16U: "u2", cv2.CV_16S: "i2",
            cv2.CV_32S: "i4",
            cv2.CV_32F: "f4", cv2.CV_64F: "f8",
        }
        size_map = {
            cv2.CV_8U: 1, cv2.CV_8S: 1,
            cv2.CV_16U: 2, cv2.CV_16S: 2,
            cv2.CV_32S: 4,
            cv2.CV_32F: 4, cv2.CV_64F: 8,
        }
        elem_size = size_map[gpu_mat.depth()]
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": (c, h, w),
            "data": (gpu_mat.cudaPtr(), False),
            "typestr": type_map[gpu_mat.depth()],
            "strides": (elem_size , gpu_mat.step, elem_size * c),
        }

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif isinstance(data, cv2.cuda_GpuMat):
        return torch.tensor(CudaArrayInterface(data), device="cuda")
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@COMMON_PIPELINES.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if not isinstance(img, cv2.cuda_GpuMat):
                # all require transpose(2, 0, 1) except cv2.cuda_GpuMat
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = img.transpose(2, 0, 1)
            results[key] = to_tensor(img)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@COMMON_PIPELINES.register_module()
class DefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and (not isinstance(img, cv2.cuda_GpuMat) and img.dtype == np.uint8):
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if isinstance(img, cv2.cuda_GpuMat):
                pass
            elif isinstance(img, torch.Tensor):
                img = img.permute(2, 0, 1).contiguous()
            else:
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))    
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        if isinstance(img, cv2.cuda_GpuMat):
            results.setdefault('pad_shape', (img.size()[1],) + (img.size()[0],) + (img.channels(),))
            num_channels = img.channels()
        else:
            results.setdefault('pad_shape', img.shape)
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault('scale_factor', 1.0)
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'