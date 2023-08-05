from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
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

def countLayers(model, desc=''):
    nb_layers = 0
    if desc == '':
        for name, m in model.named_modules():
            if len(list(m.named_children())) > 1:
                nb_layers += 1
        nb_layers -= 2 
    else:
        for name, m in model.named_modules():
            if len(list(m.named_modules())) != 1:
                nb_layers += 1
        nb_layers += 2
    print(f'The {desc} mobilenet v3 small baseline contains {nb_layers} layers.')
           
def showLayersInfo(trunk, extra_layers):
    countLayers(trunk)       
    countLayers(extra_layers, 'extracted') 
     
class MobileNetV3Small(nn.Module):
    def __init__(self, width_mult):
        super().__init__()
        
        if width_mult == 1:
            trunk = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, width_mult=width_mult)
        else:
            trunk = mobilenet_v3_small(width_mult=width_mult)
        
        self.trunk = create_feature_extractor(
            trunk,
            return_nodes={
                'features.8.block.0': 'C4',
                'features.12': 'C5',
            }
        )

        self.extra_layers = nn.ModuleList(
            [
                _ExtraBlock(576, 288),
                _ExtraBlock(288, 144),
                _ExtraBlock(144, 144),
                _ExtraBlock(144, 72)
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
