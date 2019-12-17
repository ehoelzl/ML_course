""" Full assembly of the parts to form the complete network """
import numpy as np

from torch.nn import ModuleList
from torch_unet.unet.components import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth, init_filters=16, padding=False, batch_norm=False, dropout=0.,
                 leaky=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.padding = padding
        self.depth = depth
        
        self.down_path = ModuleList()
        self.up_path = ModuleList()
        
        # Encoder
        prev_channels = n_channels
        out_channels = init_filters
        for i in range(depth - 1):
            self.down_path.append(Down(in_channels=prev_channels, out_channels=out_channels,
                                       padding=padding, batch_norm=batch_norm, leaky=leaky))
            prev_channels = out_channels
            out_channels *= 2
        
        # Bridge
        self.center = DoubleConv(in_channels=prev_channels, out_channels=out_channels, padding=padding,
                                 batch_norm=batch_norm, leaky=leaky, dropout=dropout)
        
        # Decoder
        prev_channels = out_channels
        out_channels = int(prev_channels / 2)
        for i in reversed(range(depth - 1)):
            self.up_path.append(Up(prev_channels, out_channels, padding, batch_norm, leaky=leaky))
            prev_channels = out_channels
            out_channels = int(out_channels / 2)
        
        self.out = OutConv(prev_channels, n_classes)
    
    def forward(self, x):
        bridges = []
        for i, layer in enumerate(self.down_path):
            x, bridge = layer(x)
            bridges.append(bridge)
        
        x = self.center(x)
        for i, up in enumerate(self.up_path):
            x = up(x, bridges[-i - 1])
        
        return self.out(x)


def predict_full_image(net, img, device):
    img = np.transpose(img, (2, 0, 1))  # Transpose to get (n, c, h, w)
    img = img[None, :, :, :]
    img = torch.from_numpy(img).to(device=device, dtype=torch.float32)
    
    prediction = torch.sigmoid(net(img)).cpu().detach().numpy().squeeze(1)
    return prediction
