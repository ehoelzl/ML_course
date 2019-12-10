""" Full assembly of the parts to form the complete network """

from torch.nn import ModuleList
from torch_unet.unet.components import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth, init_filters=6, padding=False, batch_norm=False, dropout=0.):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.padding = padding
        self.depth = depth
        
        self.down_path = ModuleList()
        self.up_path = ModuleList()
        
        prev_channels = n_channels
        for i in range(depth):
            out_channels = 2 ** (init_filters + i)
            if i != depth - 1:
                self.down_path.append(Down(prev_channels, out_channels, padding, batch_norm))
            else:
                self.down_path.append(DoubleConv(prev_channels, out_channels, padding, batch_norm, dropout=dropout))
            prev_channels = out_channels
        
        for i in reversed(range(depth - 1)):
            self.up_path.append(Up(prev_channels, 2 ** (init_filters + i), padding, batch_norm))
            prev_channels = 2 ** (init_filters + i)
        
        self.out = OutConv(prev_channels, n_classes)
    
    def forward(self, x):
        blocks = []
        for i, layer in enumerate(self.down_path):
            if i != len(self.down_path) - 1:
                x, conv = layer(x)
                blocks.append(conv)
            else:
                x = layer(x)
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        
        return self.out(x)
