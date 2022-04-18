import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SSCDl(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(SSCDl, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
            
        initialize_weights(self.classifier1, self.classifier2, self.resCD, self.classifierCD)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        
        x = self.FCN.layer0(x) #size:1/4
        x = self.FCN.maxpool(x) #size:1/4
        x = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x) #size:1/8
        x = self.FCN.layer3(x) #size:1/16
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)        
        return x
    
    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        change = self.CD_forward(x1, x2)
                
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        
        return F.upsample(change, x_size[2:], mode='bilinear'), F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear')