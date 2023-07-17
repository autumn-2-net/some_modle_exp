import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    def __init__(self,  period,kernel_size,stride,lReLU_slope):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = lReLU_slope
        self.period = period

        kernel_size = kernel_size
        stride = stride
        # norm_f = weight_norm if hp.mpd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0)),
            nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0)),
           nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0)),
           nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0)),
          nn.Conv2d(512, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            # fmap.append(x)
        x = self.conv_post(x)
        # fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return  x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2,3,5,7,11],kernel_size=5,stride=3,lReLU_slope=0.2):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period= period,kernel_size=kernel_size,stride=stride,lReLU_slope=lReLU_slope) for period in periods]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
