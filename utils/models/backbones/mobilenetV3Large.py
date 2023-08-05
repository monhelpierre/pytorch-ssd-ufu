from torch import nn
from torchvision.models import mobilenet_v3_large
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

def showLayersInfo(trunk, extra_layers):
    nb_layers = 0
    for name, m in trunk.named_modules():
        nb_layers += 1
    print(f'The mobilenet v2 baseline contains {nb_layers} layers.')
               
    nb_layers = 0
    for name, m in extra_layers.named_modules():
        nb_layers += 1
    print(f'The extracted mobilenet v2 baseline contains {nb_layers} layers.')
      
class MobileNetV3Large(nn.Module):
    def __init__(self, width_mult):
        super().__init__()
        trunk = mobilenet_v3_large(pretrained=(width_mult == 1), width_mult=width_mult)
        self.trunk = create_feature_extractor(
            trunk,
            return_nodes={
                'features.14.block.0': 'C4',
                'features.16': 'C5',
            }
        )

        self.extra_layers = nn.ModuleList(
            [
                _ExtraBlock(960, 480),
                _ExtraBlock(480, 240),
                _ExtraBlock(240, 240),
                _ExtraBlock(240, 120),
            ]
        )
        
        showLayersInfo(trunk, self.extra_layers)

    def forward(self, images):
        ftrs = self.trunk(images)
        C4 = ftrs['C4']
        C5 = ftrs['C5']
        C6 = self.extra_layers[0](C5)
        C7 = self.extra_layers[1](C6)
        C8 = self.extra_layers[2](C7)
        C9 = self.extra_layers[3](C8)
        return [C4, C5, C6, C7, C8, C9]
