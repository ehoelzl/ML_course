""" Parts of the U-Net model """

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, activation=True):
        super(DoubleConv, self).__init__()
        
        block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding))]
        
        if activation:
            block.append(nn.ReLU())
        
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        if activation:
            block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        
        self.double_conv = nn.Sequential(*block)
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm=False):
        super(Down, self).__init__()
        
        self.conv = DoubleConv(in_channels, out_channels, padding, batch_norm)
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        conved = self.conv(x)
        return self.max_pool(conved), conved


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels, padding, batch_norm, activation=True)
    
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]
    
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv(out)
        
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
