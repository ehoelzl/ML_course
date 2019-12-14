""" Full assembly of the parts to form the complete network """
import numpy as np

from torch.nn import ModuleList
from torch_unet.unet.components import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth, init_filters=6, padding=False, batch_norm=False, dropout=0.,
                 leaky=False):
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
                # Only add dilation after Max pool layers
                self.down_path.append(Down(in_channels=prev_channels, out_channels=out_channels,
                                           padding=padding, batch_norm=batch_norm, leaky=leaky))
            else:
                # Add dropout at end of contracting path (according to paper)
                self.down_path.append(DoubleConv(in_channels=prev_channels, out_channels=out_channels,
                                                 padding=padding, batch_norm=batch_norm, dropout=dropout, leaky=leaky))
            prev_channels = out_channels
        
        for i in reversed(range(depth - 1)):
            self.up_path.append(Up(prev_channels, 2 ** (init_filters + i), padding, batch_norm, leaky=leaky))
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


def predict_full_image(net, img, device):
    img = np.transpose(img, (2, 0, 1))  # Transpose to get (n, c, h, w)
    img = img[None, :, :, :]
    img = torch.from_numpy(img).to(device=device, dtype=torch.float32)
    
    prediction = torch.sigmoid(net(img)).cpu().detach().numpy().squeeze(1)
    return prediction
