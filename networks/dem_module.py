import math
from re import S
from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F


class DEMBlock(nn.Module):
    def __init__(self, channel, ssp_level):
        super(DEMBlock, self).__init__()
        self.channel = channel
        self.ssp_level = ssp_level
        
        ssp_block_num = 0
        for ssp in ssp_level:
            ssp_block_num += int(math.pow(2, ssp-1))**2

        self.conv_local = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1)
        self.fc_global = nn.Linear(ssp_block_num*self.channel, self.channel)
        self.conv_weightL = nn.Conv2d(self.channel*2, self.channel, 1)
        self.conv_weightG = nn.Conv2d(self.channel*2, self.channel, 1)

    def forward(self, feature_I, feature_L):

        refined_I = self._forward(feature_I, feature_L)
        refined_L = self._forward(feature_I, feature_L)

        return refined_I, refined_L

    def _forward(self, input, aux):
        _, c, h, w = input.shape

        local_info = self.conv_local(aux)

        info_vector = []
        for ssp in self.ssp_level:
            region_dim = int(math.pow(2, ssp-1))
            chunks = []
            for chunk_h in torch.chunk(local_info, region_dim, dim=2):
                for chunk_w in torch.chunk(chunk_h, region_dim, dim=3):
                    chunks.append(chunk_w)
            
            for chunk in chunks:
                info_vector.append(F.max_pool2d(chunk, (int(h/region_dim), int(w/region_dim))))

        global_vector = torch.cat(info_vector, 1)
        global_vector = self.fc_global(global_vector.squeeze(2).squeeze(2))
        global_info = global_vector.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

        total_info = torch.cat((local_info, global_info), dim=1)

        weightL = torch.sigmoid(self.conv_weightL(total_info))
        weightG = torch.sigmoid(self.conv_weightG(total_info))

        refined = input + local_info * weightL + global_info * weightG

        return refined

class DEMBlock_gconv(nn.Module):
    def __init__(self, dim, ssp_level, gconv):
        super(DEMBlock_gconv, self).__init__()
        self.dim = dim
        self.ssp_level = ssp_level
        
        ssp_block_num = 0
        for ssp in ssp_level:
            ssp_block_num += int(math.pow(2, ssp-1))**2

        if gconv == "decoder":
            self.conv_local = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        else:
            self.conv_local = gconv(self.dim)
            
        self.fc_global = nn.Linear(ssp_block_num*self.dim, self.dim)
        self.conv_weightL = nn.Conv2d(self.dim*2, self.dim, 1)
        self.conv_weightG = nn.Conv2d(self.dim*2, self.dim, 1)

    def forward(self, feature_I, feature_L):

        refined_I = self._forward(feature_I, feature_L)
        refined_L = self._forward(feature_I, feature_L)

        return refined_I, refined_L

    def _forward(self, input, aux):
        _, c, h, w = input.shape

        local_info = self.conv_local(aux)

        info_vector = []
        for ssp in self.ssp_level:
            region_dim = int(math.pow(2, ssp-1))
            chunks = []
            for chunk_h in torch.chunk(local_info, region_dim, dim=2):
                for chunk_w in torch.chunk(chunk_h, region_dim, dim=3):
                    chunks.append(chunk_w)
            
            for chunk in chunks:
                info_vector.append(F.max_pool2d(chunk, (int(h/region_dim), int(w/region_dim))))

        global_vector = torch.cat(info_vector, 1)
        global_vector = self.fc_global(global_vector.squeeze(2).squeeze(2))
        global_info = global_vector.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

        total_info = torch.cat((local_info, global_info), dim=1)

        weightL = torch.sigmoid(self.conv_weightL(total_info))
        weightG = torch.sigmoid(self.conv_weightG(total_info))

        refined = input + local_info * weightL + global_info * weightG
        
        return refined