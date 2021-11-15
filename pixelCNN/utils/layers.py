import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

from utils.helpers import *


class DownShiftedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=(2,3), stride=(1,1),
                 shift_output_down=False, norm='weight_norm'):
        super(DownShiftedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm

        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # left
                                int((filter_size[1] - 1) / 2),  # right
                                filter_size[0] - 1,             # up
                                0) )                            # down
        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(out_channels)
        
        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0,0,1,0)))
            
    def forward(self, x):
        # Forward pass
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class DownRightShiftedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=(2,2), stride=(1,1),
                 shift_output_right=False, norm='weight_norm'):
        super(DownRightShiftedConv2d, self).__init__()
        
        self.pad = nn.ZeroPad2d((filter_size[1]-1, 0, filter_size[0]-1, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride)
        self.shift_output_right = shift_output_right
        self.norm = norm
        
        if norm =='weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(out_channels)
        
        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1,0,0,0)))
            
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class DownShiftedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=(2,3), stride=(1,1)):
        super(DownShiftedDeconv2d, self).__init__()
        
        self.deconv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, filter_size, stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride
        
    def forward(self, x):
        # Forward pass
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1)/2):(xs[3] - int((self.filter_size[1] - 1)/2))]

class DownRightShiftedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=(2,2), stride=(1,1)):
        super(DownRightShiftedDeconv2d, self).__init__()

        self.deconv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, filter_size,
                                            stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        # Forward pass
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


class NiN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NiN, self).__init__()
        self.lin_a = weight_norm(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out
    
    def forward(self, x):
        # forward pass: 1x1 convolution
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3])) # y = xA^T + b in torch
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class GatedResnet(nn.Module):
    def __init__(self, num_filters, conv_op, skip_connection=0):
        super(GatedResnet, self).__init__()
        self.skip_connection = skip_connection
        self.conv_input = conv_op(2*num_filters, num_filters) # concatenated ELU doubles channel dimension
        
        if skip_connection != 0:
            self.nin = NiN(2*skip_connection*num_filters, num_filters)
        
        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2*num_filters, 2*num_filters)
        
    def forward(self, x_in, x_skip=None):
        # Forward pass
        x = self.conv_input(concat_elu(x_in))
        if x_skip is not None:
            x += self.nin(concat_elu(x_skip))
        x = concat_elu(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        x_out = a*torch.sigmoid(b)
        return x_in + x_out