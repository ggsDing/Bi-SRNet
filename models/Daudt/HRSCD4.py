# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# rcdaudt.github.io
#
# If you use this code in your work, please cite the reference below:
# Daudt, Rodrigo Caye, et al. "Multitask learning for large-scale semantic change detection." Computer Vision and Image Understanding (2019).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_ss(nn.Module):

    def __init__(self, inplanes, planes = None, subsamp=1):
        super(BasicBlock_ss, self).__init__()
        if planes == None:
            planes = inplanes * subsamp
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.subsamp = subsamp
        self.doit = planes != inplanes
        if self.doit:
            self.couple = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        if self.doit:
            residual = self.couple(x)
            residual = self.bnc(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.subsamp > 1:
            out = F.max_pool2d(out, kernel_size=self.subsamp, stride=self.subsamp)
            residual = F.max_pool2d(residual, kernel_size=self.subsamp, stride=self.subsamp)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)

        return out
    

    
class BasicBlock_us(nn.Module):

    def __init__(self, inplanes, upsamp=1):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp) # assumes integer result, fix later
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1, bias=False) 
        self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """Encoder block"""
    def __init__(self, input_nbr):
        """Init Encoder fields."""
        super(Encoder, self).__init__()

        self.input_nbr = input_nbr
        
        
        cur_depth = input_nbr
        
        base_depth = 8
        
        # Encoding stage 1
        self.encres1_1 = BasicBlock_ss(cur_depth, planes = base_depth)
        cur_depth = base_depth
        d1 = base_depth
        self.encres1_2 = BasicBlock_ss(cur_depth)
        self.encres1_3 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 2
        self.encres2_1 = BasicBlock_ss(cur_depth)
        d2 = cur_depth
        self.encres2_2 = BasicBlock_ss(cur_depth)
        self.encres2_3 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 3
        self.encres3_1 = BasicBlock_ss(cur_depth)
        d3 = cur_depth
        self.encres3_2 = BasicBlock_ss(cur_depth)
        self.encres3_3 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 4
        self.encres4_1 = BasicBlock_ss(cur_depth)
        d4 = cur_depth
        self.encres4_2 = BasicBlock_ss(cur_depth)
        self.encres4_3 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 5
        self.encres5_1 = BasicBlock_ss(cur_depth)
        d5 = cur_depth
        self.encres5_2 = BasicBlock_ss(cur_depth)
        self.encres5_3 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Depths
        self.depths = [input_nbr, d1, d2, d3, d4, d5, cur_depth]
        
        
    def forward(self, x):
                
        s1_1 = x.size()
        x = self.encres1_1(x)
        x1 = self.encres1_2(x)
        x = self.encres1_3(x1)
        
        s2_1 = x.size()
        x = self.encres2_1(x)
        x2 = self.encres2_2(x)
        x = self.encres2_3(x2)
        
        s3_1 = x.size()
        x = self.encres3_1(x)
        x3 = self.encres3_2(x)
        x = self.encres3_3(x3)
        
        s4_1 = x.size()
        x = self.encres4_1(x)
        x4 = self.encres4_2(x)
        x = self.encres4_3(x4)
        
        s5_1 = x.size()
        x = self.encres5_1(x)
        x5 = self.encres5_2(x)
        x = self.encres5_3(x5)
        
        sizes = list()
        sizes.append(s1_1)
        sizes.append(s2_1)
        sizes.append(s3_1)
        sizes.append(s4_1)
        sizes.append(s5_1)
        
        outputs = list()
        outputs.append(x1)
        outputs.append(x2)
        outputs.append(x3)
        outputs.append(x4)
        outputs.append(x5)
        outputs.append(x)
        
        return outputs, sizes
    

class Decoder(nn.Module):
    """Decoder block"""
    def __init__(self, label_nbr, depths, CD = False):
        """Init Decoder fields."""
        super(Decoder, self).__init__()
        
        cur_depth = depths[6]
        
        # Decoding stage 5
        self.decres5_1 = BasicBlock_ss(cur_depth + CD * depths[6], planes = cur_depth)
        self.decres5_2 = BasicBlock_ss(cur_depth)
        self.decres5_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = depths[5]
        
        # Decoding stage 4
        self.decres4_1 = BasicBlock_ss(cur_depth + depths[5] + CD * depths[5], planes = cur_depth)
        self.decres4_2 = BasicBlock_ss(cur_depth)
        self.decres4_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = depths[4]
        
        # Decoding stage 3
        self.decres3_1 = BasicBlock_ss(cur_depth + depths[4] + CD * depths[4], planes = cur_depth)
        self.decres3_2 = BasicBlock_ss(cur_depth)
        self.decres3_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = depths[3]
        
        # Decoding stage 2
        self.decres2_1 = BasicBlock_ss(cur_depth + depths[3] + CD * depths[3], planes = cur_depth)
        self.decres2_2 = BasicBlock_ss(cur_depth)
        self.decres2_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = depths[2]
        
        # Decoding stage 1
        self.decres1_1 = BasicBlock_ss(cur_depth + depths[2] + CD * depths[2], planes = cur_depth)
        self.decres1_2 = BasicBlock_ss(cur_depth)
        self.decres1_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = depths[1]
        
        # Decoding stage 0
        self.decres0_1 = BasicBlock_ss(cur_depth + depths[1] + CD * depths[1], planes = cur_depth)
        self.decres0_2 = BasicBlock_ss(cur_depth)
        self.decres0_3 = BasicBlock_ss(cur_depth)
        
        # Output
        #self.coupling = nn.Conv2d(cur_depth + depths[1] + CD * depths[1], label_nbr, kernel_size=1)
        self.coupling = nn.Conv2d(cur_depth, label_nbr, kernel_size=1)
        #self.sm = nn.LogSoftmax(dim=1)
        
    def forward(self, outputs, sizes):
        x = self.decres5_1(outputs[5])
        x = self.decres5_2(x)
        x = self.decres5_3(x)
        s5_1 = sizes[4]
        s5_2 = x.size()
        pad5 = ReplicationPad2d((0, s5_1[3] - s5_2[3], 0, s5_1[2] - s5_2[2]))
        x = pad5(x)
        
        x = self.decres4_1(torch.cat((x, outputs[4]), 1))
        x = self.decres4_2(x)
        x = self.decres4_3(x)
        s4_1 = sizes[3]
        s4_2 = x.size()
        pad4 = ReplicationPad2d((0, s4_1[3] - s4_2[3], 0, s4_1[2] - s4_2[2]))
        x = pad4(x)
        
        x = self.decres3_1(torch.cat((x, outputs[3]), 1))
        x = self.decres3_2(x)
        x = self.decres3_3(x)
        s3_1 = sizes[2]
        s3_2 = x.size()
        pad3 = ReplicationPad2d((0, s3_1[3] - s3_2[3], 0, s3_1[2] - s3_2[2]))
        x = pad3(x)
        
        x = self.decres2_1(torch.cat((x, outputs[2]), 1))
        x = self.decres2_2(x)
        x = self.decres2_3(x)
        s2_1 = sizes[1]
        s2_2 = x.size()
        pad2 = ReplicationPad2d((0, s2_1[3] - s2_2[3], 0, s2_1[2] - s2_2[2]))
        x = pad2(x)
        
        x = self.decres1_1(torch.cat((x, outputs[1]), 1))
        x = self.decres1_2(x)
        x = self.decres1_3(x)
        s1_1 = sizes[0]
        s1_2 = x.size()
        pad1 = ReplicationPad2d((0, s1_1[3] - s1_2[3], 0, s1_1[2] - s1_2[2]))
        x = pad1(x)
        
        x = self.decres0_1(torch.cat((x, outputs[0]), 1))
        x = self.decres0_2(x)
        x = self.decres0_3(x)
        
        x = self.coupling(x)
        #x = self.sm(x)
        
        return x

    
class HRSCD4(nn.Module):
    def __init__(self, input_nbr, label_nbr, wsl = False):
        super(HRSCD4, self).__init__()

        self.input_nbr = input_nbr
        self.label_nbr = label_nbr
        
        # Terrain Classification
        self.TCEncoder = Encoder(self.input_nbr)
        self.TCDecoder = Decoder(self.label_nbr, self.TCEncoder.depths)
        
        # Change Detection
        self.CDEncoder = Encoder(2 * self.input_nbr)
        if wsl:
            self.CDDecoder = Decoder(2, self.CDEncoder.depths, CD = True)
        else:
            self.CDDecoder = Decoder(1, self.CDEncoder.depths, CD = True)
        
    def forward(self, x1, x2):

        #x = torch.cat((x1, x2), 1)
        
        # Terrain Classification - Image 1
        outputs_1, sizes_1 = self.TCEncoder(x1)
        tc1 = self.TCDecoder(outputs_1, sizes_1)
        
        # Terrain Classification - Image 2
        outputs_2, sizes_2 = self.TCEncoder(x2)
        tc2 = self.TCDecoder(outputs_2, sizes_2)
        
        # Change Detection
        outputs_cd, sizes_cd = self.CDEncoder(torch.cat((x1, x2), 1))
        for i in range(len(outputs_cd)):
            outputs_cd[i] = torch.cat((outputs_cd[i], torch.abs(outputs_1[i] - outputs_2[i])), 1)
        cm = self.CDDecoder(outputs_cd, sizes_cd)
        
        return cm, tc1, tc2
