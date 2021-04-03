from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from . import BaseNet


__all__ = ['DFF', 'get_dff']
class ChannelAttention(nn.Module):
    def __init__(self, k_size=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #
        # self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        #
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out

        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        ym = self.max_pool(x)
        ym = self.conv(ym.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y =y+ym
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return y
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class DFF(BaseNet):
    r"""Dynamic Feature Fusion for Semantic Edge Detection

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Yuan Hu, Yunpeng Chen, Xiang Li, Jiashi Feng. "Dynamic Feature Fusion 
        for Semantic Edge Detection" *IJCAI*, 2019

    """
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DFF, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass

        self.ada_learner = LocationAdaptiveLearner(nclass, nclass*4, nclass*4, norm_layer=norm_layer)


        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(64, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(128, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(512, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))

        self.side5_w = nn.Sequential(nn.Conv2d(512, nclass*4, 1, bias=True),
                                   norm_layer(nclass*4),
                                   nn.ConvTranspose2d(nclass*4, nclass*4, 16, stride=8, padding=4, bias=False))

        self.canny_layer = nn.Sequential(nn.Conv2d(1, 1, 1))
        # self.con_layer1 = nn.Sequential(nn.Conv2d(64, 64, 1))
        # self.con_layer2 = nn.Sequential(nn.Conv2d(64, 64, 1))
        # self.con_layer3 = nn.Sequential(nn.Conv2d(128, 128, 1))
        # self.con_layer4 = nn.Sequential(nn.Conv2d(512, 512, 1))
        # self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
        #                            norm_layer(1))
        # self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
        #                            norm_layer(1),
        #                            nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        # self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
        #                            norm_layer(1),
        #                            nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        # self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
        #                            norm_layer(nclass),
        #                            nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        #
        # self.side5_w = nn.Sequential(nn.Conv2d(2048, nclass*4, 1, bias=True),
        #                            norm_layer(nclass*4),
        #                            nn.ConvTranspose2d(nclass*4, nclass*4, 16, stride=8, padding=4, bias=False))
        self.ca1 = ChannelAttention()
        self.sa1 = SpatialAttention()


        self.ca2 = ChannelAttention()
        self.sa2 = SpatialAttention()


        self.ca3 = ChannelAttention()
        self.sa3 = SpatialAttention()

        self.ca5 = ChannelAttention()
        self.sa5 = SpatialAttention()
    def forward(self, x,canny_edge):
        c1, c2, c3, _, c5 = self.base_forward(x)


        c11 = self.ca1(c1)*c1
        c12 = self.sa1(c1)*c1
        c1 = c11+c12+c1
        # c1 = self.con_layer1(c1)
        #
        c21 = self.ca2(c2) * c2
        c22 = self.sa2(c2) * c2
        c2 = c21 + c22+c2
        # c2 = self.con_layer2(c2)
        #
        c31 = self.ca3(c3) * c3
        c32 = self.sa3(c3) * c3
        c3 = c31 + c32+c3
        # c3 = self.con_layer3(c3)
        #
        c51 = self.ca5(c5) * c5
        c52 = self.sa5(c5) * c5
        c5 = c51 + c52+c5
        # c5 = self.con_layer4(c5)

        side1 = self.side1(c1) # (N, 1, H, W)
        side2 = self.side2(c2) # (N, 1, H, W)
        side3 = self.side3(c3) # (N, 1, H, W)
        side5 = self.side5(c5) # (N, 19, H, W)
        side5_w = self.side5_w(c5) # (N, 19*4, H, W)
        
        ada_weights = self.ada_learner(side5_w) # (N, 19, 4, H, W)

        slice5 = side5[:,0:1,:,:] # (N, 1, H, W)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        # for i in range(side5.size(1)-1):
        #     slice5 = side5[:,i+1:i+2,:,:] # (N, 1, H, W)
        #     fuse = torch.cat((fuse, slice5, side1, side2, side3), 1) # (N, 19*4, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3)) # (N, 19, 4, H, W)
        fuse = torch.mul(fuse, ada_weights) # (N, 19, 4, H, W)
        fuse = torch.sum(fuse, 2) # (N, 19, H, W)
        # B = side5[0].tolist()
        # B= np.array(np.float64(B))
        # A =np.array(fuse[0].tolist())

        side5 = torch.cat((side5,canny_edge),1)
        side5 = torch.sum(side5, 1)
        side5 = side5.view(side5.size(0), self.nclass, side5.size(1), side5.size(2))
        fuse = torch.cat((fuse,canny_edge),1)
        fuse = torch.sum(fuse, 1)
        fuse = fuse.view(fuse.size(0), self.nclass, fuse.size(1), fuse.size(2))
        side5 = self.canny_layer(side5)
        fuse = self.canny_layer(fuse)
        # A = np.array(fuse[0].tolist())
        # B = np.array(side5[0].tolist())
        outputs = [side5, fuse]

        return tuple(outputs)


class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x) # (N, 19*4, H, W)
        x = self.conv2(x) # (N, 19*4, H, W)
        x = self.conv3(x) # (N, 19*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3)) # (N, 19, 4, H, W)
        return x


def get_dff(dataset='cityscapes', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""DFF model from the paper "Dynamic Feature Fusion for Semantic Edge Detection"
    """
    acronyms = {
        'cityscapes': 'cityscapes',
        'sbd': 'sbd',
    }
    # infer number of classes
    from datasets import datasets
    model = DFF(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

