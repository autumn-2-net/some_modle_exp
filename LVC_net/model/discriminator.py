import torch
import torch.nn as nn

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
# from omegaconf import OmegaConf

class Discriminator(nn.Module):
    def __init__(self,mpd_periods=[2,3,5,7,11],mpd_kernel_size=5,mpd_stride=3,mpd_lReLU_slope=0.2,mrd_resolutions=[(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)],mrd_lReLU_slope=0.2):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(resolutions=mrd_resolutions,lReLU_slope=mrd_lReLU_slope)
        self.MPD = MultiPeriodDiscriminator(periods=mpd_periods,kernel_size=mpd_kernel_size,stride=mpd_stride,lReLU_slope=mpd_lReLU_slope)

    def forward(self, x):
        return self.MRD(x), self.MPD(x)
#
# if __name__ == '__main__':
#     hp = OmegaConf.load('../config/default.yaml')
#     model = Discriminator(hp)
#
#     x = torch.randn(3, 1, 16384)
#     print(x.shape)
#
#     mrd_output, mpd_output = model(x)
#     for features, score in mpd_output:
#         for feat in features:
#             print(feat.shape)
#         print(score.shape)
#
#     pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)

