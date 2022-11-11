"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import math
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial


nonlinearity = partial(F.relu,inplace=True)


class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPDblock(nn.Module):
    def __init__(self,channel):
        super(SPDblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        
        self.spm = SPBlock(channel, channel, norm_layer=nn.BatchNorm2d)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        spm_out = self.spm(x)
        out = (x + dilate1_out + dilate2_out + dilate3_out + dilate4_out) * spm_out
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
#         norm_layer = nn.BatchNorm2d
#         up_kwargs = {'mode': 'bilinear', 'align_corners': True}
#         self.strip_pool1 = StripPooling(512, (20, 12), norm_layer, up_kwargs)
#         self.strip_pool2 = StripPooling(512, (20, 12), norm_layer, up_kwargs)
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        _, _, h, w = x.size()
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
#         e4 = self.strip_pool1(e4)
#         e4 = self.strip_pool2(e4)
        e4 = self.dblock(e4)
 
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
    
class DinkNet34_SP(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34_SP, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = resnet34()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        norm_layer = nn.BatchNorm2d
        up_kwargs = {'mode': 'bilinear', 'align_corners': True}
#         self.strip_pool1 = StripPooling(512, (20, 12), norm_layer, up_kwargs)
#         self.strip_pool2 = StripPooling(512, (20, 12), norm_layer, up_kwargs)
#        self.strip_pool3 = StripPooling_multi(512, (20, 12), norm_layer, up_kwargs)
#        self.strip_pool4 = StripPooling_multi(512, (20, 12), norm_layer, up_kwargs)
        
        self.dblock = Dblock(512)
#        self.spdblock = SPDblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        _, _, h, w = x.size()
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        #print value of e4, form (B,C,H,W)
        #e4n = e4.cpu().detach().numpy()
        #print("[30,30]= ", e4n[1][30][30][30], file=mylogse)

        # Center
#         e4 = self.strip_pool1(e4) 
#         e4 = self.strip_pool2(e4)
        e4 = self.dblock(e4)
#        e4 = self.spdblock(e4)
#        e4 = self.strip_pool3(e4)
#        e4 = self.strip_pool4(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
        
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        ############################
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        ############################
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        # self.pool3 = nn.AdaptiveAvgPool2d((None, 2))
        # self.pool4 = nn.AdaptiveAvgPool2d((2, None))
        # self.pool5 = nn.AdaptiveAvgPool2d((None, 3))
        # self.pool6 = nn.AdaptiveAvgPool2d((3, None))
#         self.pool7 = nn.AdaptiveAvgPool2d((None, 4))
#         self.pool8 = nn.AdaptiveAvgPool2d((4, None))
        ############################
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        #print(x.shape)
        _, _, h, w = x.size()
        self.pool3 = nn.AdaptiveAvgPool2d((1, w-1))
        self.pool4 = nn.AdaptiveAvgPool2d((h-1, 1))
        self.pool5 = nn.AdaptiveAvgPool2d((1, w-2))
        self.pool6 = nn.AdaptiveAvgPool2d((h-2, 1))
        x1 = self.pool1(x)
        #print(x1.shape)
        x1 = self.conv1(x1)
        #print(x1.shape)
        x1 = self.bn1(x1)
        #print(x1.shape)
        x1 = x1.expand(-1, -1, h, w)
        #print(x1.shape)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x3 = self.pool3(x)
        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = x3.expand(-1, -1, h, w-1)
        x3 = F.interpolate(x3, (h, w))

        x4 = self.pool4(x)
        x4 = self.conv2(x4)
        x4 = self.bn2(x4)
        x4 = x4.expand(-1, -1, h-1, w)
        x4 = F.interpolate(x4, (h, w))

        x5 = self.pool5(x)
        x5 = self.conv1(x5)
        x5 = self.bn1(x5)
        x5 = x5.expand(-1, -1, h, w-2)
        x5 = F.interpolate(x5, (h, w))

        x6 = self.pool6(x)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = x6.expand(-1, -1, h-2, w)
        x6 = F.interpolate(x6, (h, w))
        
#         x7 = self.pool7(x)
#         x7 = self.conv1(x7)
#         x7 = self.bn1(x7)
#         #x5 = x5.expand(-1, -1, h, w)
#         x7 = F.interpolate(x7, (h, w))

#         x8 = self.pool8(x)
#         x8 = self.conv2(x8)
#         x8 = self.bn2(x8)
#         #x6 = x6.expand(-1, -1, h, w)
#         x8 = F.interpolate(x8, (h, w))
        
        x = self.relu(x1 + x2 + x3 + x4 + x5 + x6)
        x = self.conv3(x).sigmoid()
        return x



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None, spm_on=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.spm = None
        if spm_on:
            self.spm = SPBlock(planes, planes, norm_layer=norm_layer)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.spm is not None:
            out = out * self.spm(out) #add SPM after the first Conv3x3
            print("this does work.")

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        
        spm_on = False
        if planes == 512:
            spm_on = True

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=4, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i >= blocks - 1 or planes == 512:
                spm_on = True
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    # model.load_state_dict(torch.load('weights/resnet34_pytorch.pth'), strict=False)
    return model
 