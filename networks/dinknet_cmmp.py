import re
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial
from networks.dinknet import Dblock, DecoderBlock, DinkNet34
from networks.dem_module import DEMBlock


nonlinearity = partial(F.relu, inplace=True)

class DinkNet34_AE(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_AE, self).__init__()

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
    
    def forward_step1(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        return x


class DinkNet34CMMP(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, ssp_level = [1, 2, 3]):
        super(DinkNet34CMMP, self).__init__()
        filters = [64, 128, 256, 512]
        channel_size = [64, 128, 256, 512, 256, 128, 64, 64]
        dim_size = [128, 64, 32, 16, 32, 64, 128, 256]
        
        self.net_I = DinkNet34_AE()
        self.net_I = self.net_I.cuda()
        self.net_L = DinkNet34_AE()
        self.net_L = self.net_L.cuda()

        self.dem_blocks = []
        for layer in range(0, len(channel_size)):
            self.dem_blocks.append(DEMBlock(channel_size[layer], ssp_level).cuda())

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.conv_fusion = nn.Conv2d(channel_size[-1]*2, channel_size[-1], 1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, input_I, input_L):

        x_I = self.net_I.forward_step1(input_I)
        x_L = self.net_L.forward_step1(input_L)

        e1_I = self.net_I.encoder1(x_I)
        e1_L = self.net_L.encoder1(x_L)
        e1_I, e1_L = self.dem_blocks[0](e1_I, e1_L)

        e2_I = self.net_I.encoder2(e1_I)
        e2_L = self.net_L.encoder2(e1_L)
        e2_I, e2_L = self.dem_blocks[1](e2_I, e2_L)

        e3_I = self.net_I.encoder3(e2_I)
        e3_L = self.net_L.encoder3(e2_L)
        e3_I, e3_L = self.dem_blocks[2](e3_I, e3_L)

        e4_I = self.net_I.encoder4(e3_I)
        e4_I = self.net_I.dblock(e4_I)
        e4_L = self.net_L.encoder4(e3_L)
        e4_L = self.net_L.dblock(e4_L)
        e4_I, e4_L = self.dem_blocks[3](e4_I, e4_L)

        d4_I = self.net_I.decoder4(e4_I) + e3_I
        d4_L = self.net_L.decoder4(e4_L) + e3_L
        d4_I, d4_L = self.dem_blocks[4](d4_I, d4_L)

        d3_I = self.net_I.decoder3(d4_I) + e2_I
        d3_L = self.net_L.decoder3(d4_L) + e2_L
        d3_I, d3_L = self.dem_blocks[5](d3_I, d3_L)

        d2_I = self.net_I.decoder2(d3_I) + e1_I
        d2_L = self.net_I.decoder2(d3_I) + e1_I
        d2_I, d2_L = self.dem_blocks[6](d2_I, d2_L)

        d1_I = self.net_I.decoder1(d2_I)
        d1_L = self.net_L.decoder1(d2_L)
        d1_I, d1_L = self.dem_blocks[7](d1_I, d1_L)
        fusion = self.conv_fusion(torch.cat((d1_I, d1_L), dim=1))

        out = self.finaldeconv1(fusion)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
