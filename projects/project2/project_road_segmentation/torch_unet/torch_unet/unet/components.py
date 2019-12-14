""" Parts of the U-Net model """

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(BN => convolution => ReLU) * 2 + Dropout"""
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, dropout=0, leaky=False):
        super(DoubleConv, self).__init__()
        
        # First conv
        block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)),
                 nn.LeakyReLU() if leaky else nn.ReLU()]
        
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        
        if dropout > 0:
            block.append(nn.Dropout2d(dropout))
        
        block += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)),
                  nn.LeakyReLU() if leaky else nn.ReLU()]
        
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        
        if dropout > 0:
            block.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*block)
    
    def forward(self, x):
        
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, leaky=False):
        super(Down, self).__init__()
        
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels,
                               padding=padding, batch_norm=batch_norm, leaky=leaky)
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        conved = self.conv(x)
        return self.max_pool(conved), conved


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, leaky=False):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels,
                               padding=padding, batch_norm=batch_norm, leaky=leaky)
    
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]
    
    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, bridge], 1)
        out = self.conv(out)
        
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
