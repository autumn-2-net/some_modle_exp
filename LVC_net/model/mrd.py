import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution,lReLU_slope):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = lReLU_slope

        # norm_f = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (3, 9), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
           nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
         nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
        ])
        self.conv_post = nn.Conv2d(32, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            # fmap.append(x)
        x = self.conv_post(x)
        # fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return  x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False,return_complex=False) #[B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self,resolutions=[(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)],lReLU_slope=0.2):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = resolutions
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution= resolution,lReLU_slope=lReLU_slope) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
