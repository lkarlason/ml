import torch
import torch.nn.functional as F
from utils.layers import *

class PixelCnnUp(torch.nn.Module):
    def __init__(self, nr_resnet, nr_filters):
        super(PixelCnnUp, self).__init__()
        self.nr_resnet = nr_resnet
        
        # Consider pixels above
        self.u_stream = torch.nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d) for _ in range(nr_resnet)])
        
        # Consider pixels above and to the left
        self.ul_stream = torch.nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, skip_connection=1) for _ in range(nr_resnet)])

    def forward(self, u, ul):
        # Forward pass: resnet blocks
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, x_skip=u)
            
            u_list += [u]
            ul_list += [ul]
        return u_list, ul_list

class PixelCnnDown(torch.nn.Module):
    def __init__(self, nr_resnet, nr_filters):
        super(PixelCnnDown, self).__init__()
        self.nr_resnet = nr_resnet
        
        # Consider pixels above
        self.u_stream = torch.nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, skip_connection=1) for _ in range(nr_resnet)])
        
        # Consider pixels above and to the left
        self.ul_stream = torch.nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, skip_connection=2) for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        # Forward pass: resnet blocks
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, x_skip=u_list.pop())
            ul = self.ul_stream[i](ul, x_skip=torch.cat((u, ul_list.pop()), 1))
        
        return u, ul

class PixelCNN(torch.nn.Module):
    def __init__(self, config):
        super(PixelCNN, self).__init__()
        self.nr_resnet = config.nr_resnet
        self.nr_filters = config.nr_filters
        self.input_channels = config.input_channels
        self.nr_logistic_mix = config.nr_logistic_mix
        
        self.u_init = DownShiftedConv2d(self.input_channels+1, self.nr_filters, shift_output_down=True)
        self.ul_init = torch.nn.ModuleList([
            DownShiftedConv2d(self.input_channels+1, self.nr_filters, filter_size=(1,3), shift_output_down=True),
            DownRightShiftedConv2d(self.input_channels+1, self.nr_filters, filter_size=(2,1), shift_output_right=True)
        ])
        
        self.up_layers = torch.nn.ModuleList([PixelCnnUp(self.nr_resnet, self.nr_filters) for i in range(3)])
        self.downsize_u = torch.nn.ModuleList([DownShiftedConv2d(self.nr_filters, self.nr_filters, stride=(2,2)) for _ in range(2)])
        self.downsize_ul = torch.nn.ModuleList([DownRightShiftedConv2d(self.nr_filters, self.nr_filters, stride=(2,2)) for _ in range(2)])

        self.nr_resnet_down = [self.nr_resnet] + [self.nr_resnet+1]*2
        self.down_layers = torch.nn.ModuleList([PixelCnnDown(self.nr_resnet_down[i], self.nr_filters) for i in range(3)])

        self.upsize_u = torch.nn.ModuleList([DownShiftedDeconv2d(self.nr_filters, self.nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_ul = torch.nn.ModuleList([DownRightShiftedDeconv2d(self.nr_filters, self.nr_filters, stride=(2,2)) for _ in range(2)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin = NiN(self.nr_filters, num_mix*self.nr_logistic_mix)
        
        self.init_padding = None
    
    def forward(self, x, sample=False):
        ## Why is this done??
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = torch.autograd.Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
        
        ## UP PASS
        x = x if sample else torch.concat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                u_list += [self.downsize_u[i](u_list[-1])]
                ul_list += [self.downsize_ul[i](ul_list[-1])]
        
        
        ## DOWN PASS      
        u = u_list.pop()
        ul = ul_list.pop()  
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            if i != 2:
                u = self.upsize_u[i](u)
                ul = self.upsize_ul[i](ul)
        x_out = self.nin(F.elu(ul))    
                    
        assert len(u_list) == len(ul_list) == 0, "Something went wrong during down pass."
        
        return x_out
