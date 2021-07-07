import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import utils.adj_mat
from config import CFG

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x_align = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, c_in, n_vertex, timestep  = x.shape
            x_align = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, n_vertex, timestep]).to(x)], dim=1)
        else:
            x_align = x
        
        return x_align


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=6):
        super(TimeBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, self.kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, self.kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, self.kernel_size))

        self.align = Align(in_channels, out_channels)

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out



class STGCNBlock_Opolka(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock_Opolka, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)



class STConvOpolkaModel(torch.nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, classes):
        super(STConvOpolkaModel, self).__init__()
    
        self.norm_adj = torch.Tensor(utils.adj_mat.get_norm_adj_mat()).cuda()
        self.block1 = STGCNBlock_Opolka(num_nodes = 21, in_channels=in_channels, spatial_channels=spatial_channels, out_channels=out_channels)
        self.block2 = STGCNBlock_Opolka(num_nodes = 21, in_channels=out_channels, spatial_channels=spatial_channels, out_channels=out_channels)
        self.block3 = STGCNBlock_Opolka(num_nodes = 21, in_channels=out_channels, spatial_channels=spatial_channels, out_channels=out_channels)
        self.last_temporal = TimeBlock(in_channels=out_channels, out_channels=out_channels)
        self.fully = nn.Linear((CFG.sequence_length - (2*2+1)*5) * out_channels * 21, CFG.num_classes) #(num blocks * 2 + 1) * (kernel size -1)

    def forward(self, x):
        x = x.permute(0,2,1,3)
        x = self.block1(x, self.norm_adj)
        x = self.block2(x, self.norm_adj)
        #x = self.block3(x, self.norm_adj)
        x = self.last_temporal(x)
        x = x.flatten(1)
        x = self.fully(x)
        return x


