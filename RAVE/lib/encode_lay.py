import torch
from torch import nn




def get_padding(kernel_size, stride=1, dilation=1, mode="centered"):
    """
    Computes 'same' padding given a kernel size, stride an dilation.

    Parameters
    ----------

    kernel_size: int
        kernel_size of the convolution

    stride: int
        stride of the convolution

    dilation: int
        dilation of the convolution

    mode: str
        either "centered", "causal" or "anticausal"
    """
    if kernel_size == 1: return (0, 0)
    p = (kernel_size - 1) * dilation + 1
    half_p = p // 2
    if mode == "centered":
        p_right = p // 2
        p_left = (p - 1) // 2
    elif mode == "causal":
        p_right = 0
        p_left = p // 2 + (p - 1) // 2
    elif mode == "anticausal":
        p_right = p // 2 + (p - 1) // 2
        p_left = 0
    else:
        raise Exception(f"Padding mode {mode} is not valid")
    return (p_left, p_right)

def get_nnp(k):
    return k//2

class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)



class Encoder(nn.Module):

    def __init__(
        self,
        data_size,
        capacity,
        latent_size,
        ratios,
        n_out,
        sample_norm,
        repeat_layers,
        recurrent_layer,  # 传入了一个模型... 不过不知道干什么的
    ):
        super().__init__()
        net = [nn.Conv1d(data_size, capacity, 7, padding=get_nnp(7))]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            if sample_norm:
                net.append(SampleNorm())
            else:
                net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                nn.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r),
                    stride=r,
                   # cumulative_delay=net[-3].cumulative_delay,
                ))

            for i in range(repeat_layers - 1):
                if sample_norm:
                    net.append(SampleNorm())
                else:
                    net.append(nn.BatchNorm1d(out_dim))
                net.append(nn.LeakyReLU(.2))
                net.append(
                    cc.Conv1d(
                        out_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3),
                        cumulative_delay=net[-3].cumulative_delay,
                    ))

        net.append(nn.LeakyReLU(.2))

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=out_dim,
                    cumulative_delay=net[-2].cumulative_delay,
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            cc.Conv1d(
                out_dim,
                latent_size * n_out,
                5,
                padding=cc.get_padding(5),
                groups=n_out,
                cumulative_delay=net[-2].cumulative_delay,
            ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return z
