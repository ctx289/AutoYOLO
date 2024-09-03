from ..builder import BACKBONES
from .hrnet import HRNet


@BACKBONES.register_module()
class HRNetFrozen(HRNet):
    """ HRnet but with frozen stages
    """
    def __init__(self,
                 extra,
                 frozen_stages=-1,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):

        super().__init__(
            extra, in_channels=in_channels,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg,
            norm_eval=norm_eval, with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            multiscale_output=multiscale_output,
            pretrained=pretrained, init_cfg=init_cfg)
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            self.norm2.eval()

            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 1:
            self.layer1.eval()
            for param in self.layer1.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 2:
            self.transition1.eval()
            self.stage2.eval()
            for m in [self.transition1, self.stage2]:
                for param in m.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 3:
            self.transition2.eval()
            self.stage3.eval()
            for m in [self.transition2, self.stage3]:
                for param in m.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 4:
            self.transition3.eval()
            self.stage4.eval()
            for m in [self.transition3, self.stage4]:
                for param in m.parameters():
                    param.requires_grad = False
