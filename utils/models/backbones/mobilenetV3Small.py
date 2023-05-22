from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor
from utils.models.layers import ConvBNReLU

class _ExtraBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        intermediate_channels = out_channels // 2
        super().__init__(
            ConvBNReLU(in_channels,
                       intermediate_channels,
                       kernel_size=1,
                       relu6=True),
            ConvBNReLU(intermediate_channels,
                       intermediate_channels,
                       kernel_size=3,
                       stride=2,
                       depthwise=True,
                       relu6=True),
            ConvBNReLU(intermediate_channels,
                       out_channels,
                       kernel_size=1,
                       relu6=True),
        )

class MobileNetV3Small(nn.Module):
    def __init__(self, width_mult):
        super().__init__()
        trunk = mobilenet_v3_small(pretrained=(width_mult == 1), width_mult=width_mult)
        self.trunk = create_feature_extractor(
            trunk,
            return_nodes={
                'features.8.block.0': 'C4',
                'features.12': 'C5',
            }
        )

        """
        _ExtraBlock(96, 512),
        _ExtraBlock(256, 128),
        _ExtraBlock(128, 64),
        _ExtraBlock(64, 32),

        """

        self.extra_layers = nn.ModuleList(
            [
                _ExtraBlock(576, 288),
                _ExtraBlock(288, 144),
                _ExtraBlock(144, 144),
                _ExtraBlock(144, 72),
            ]
        )

    def forward(self, images):
        ftrs = self.trunk(images)
        C4 = ftrs['C4']
        C5 = ftrs['C5']
        C6 = self.extra_layers[0](C5)
        C7 = self.extra_layers[1](C6)
        C8 = self.extra_layers[2](C7)
        C9 = self.extra_layers[3](C8)
        return [C4, C5, C6, C7, C8, C9]
