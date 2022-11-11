import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CFilter(nn.Module):

    def __init__(self, s_size, dim):
        super().__init__()
        self.d_size = s_size
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.ad_max = nn.AdaptiveMaxPool2d((s_size, 1))
        self.ad_avg = nn.AdaptiveAvgPool2d((s_size, 1))
        self.conv_merg = nn.Conv2d(2, 1, 1, 1)
        self.conv_down = nn.Conv2d(dim * 2, dim, 1, 1)

    def forward(self, fd, fs):

        fd = self.upsample(fd)

        fd_T = fd.transpose(3,1)

        fd_A = self.ad_avg(fd_T)
        fd_A = fd_A.transpose(3,1).contiguous()

        fd_M = self.ad_max(fd_T)
        fd_M = fd_M.transpose(3,1).contiguous()

        weights = torch.sigmoid(self.conv_merg(torch.cat((fd_A, fd_M), dim=1)))
        fd = self.conv_down(fd)
        
        output = fs * weights + fd

        return output

if __name__ == '__main__':
    cfilter = CFilter(512)
    fd = torch.randn(1, 8, 256, 256)
    fs = torch.randn(1, 8, 512, 512)
    output = cfilter(fd, fs)
    print(output.shape)

    upsample = nn.Upsample(scale_factor=(1, 1), mode='bilinear', align_corners=True)

    fd1 = upsample(fd)

    print(all(fd1 == fd))