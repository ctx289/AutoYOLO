from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import \
    DefaultFormatBundle as O_DefaultFormatBundle
from mmdet.datasets.pipelines import to_tensor

from ..builder import ATTR_PIPELINES


@ATTR_PIPELINES.register_module()
class DefaultFormatBundle(O_DefaultFormatBundle):
    def __call__(self, results):
        results = super(DefaultFormatBundle, self).__call__(results)
        key = "gt_attrs"
        results[key] = DC(to_tensor(results[key]))
        return results
