import torch
import torch.nn as nn
import torch.nn.functional as F

# from SWN import SwitchNorm1d


# class Conv1d(torch.nn.Conv1d):
#     """Conv1d module with customized initialization."""
#
#     def __init__(self, *args, **kwargs):
#         """Initialize Conv1d module."""
#         super(Conv1d, self).__init__(*args, **kwargs)
#
#     def reset_parameters(self):
#         """Reset parameters."""
#         torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
#         if self.bias is not None:
#             torch.nn.init.constant_(self.bias, 0.0)
# class Conv1d1x1(Conv1d):
#     """1x1 Conv1d with customized initialization."""
#
#     def __init__(self, in_channels, out_channels, bias):
#         """Initialize 1x1 Conv1d module."""
#         super(Conv1d1x1, self).__init__(in_channels, out_channels,
#                                         kernel_size=1, padding=0,
#                                         dilation=1, bias=bias)


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 dilation_factor=1,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 # use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                nn.Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer =  nn.Conv1d(
            conv_in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        # if use_weight_norm:
        #     self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        for f in self.conv_layers:
            x = f(x)
        return x

    # def apply_weight_norm(self):
    #     """Apply weight normalization module from all of the layers."""
    #     def _apply_weight_norm(m):
    #         if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
    #             torch.nn.utils.weight_norm(m)
    #             logging.debug(f"Weight norm is applied to {m}.")
    #
    #     self.apply(_apply_weight_norm)
    #
    # def remove_weight_norm(self):
    #     """Remove weight normalization module from all of the layers."""
    #     def _remove_weight_norm(m):
    #         try:
    #             logging.debug(f"Weight norm is removed from {m}.")
    #             torch.nn.utils.remove_weight_norm(m)
    #         except ValueError:  # this module didn't have weight norm
    #             return
    #
    #     self.apply(_remove_weight_norm)

if __name__=='__main__':
    initmd=LVCNetGenerator(in_channels=1,
                 out_channels=1,
                 inner_channels=8,
                 cond_channels=80,
                 cond_hop_length=512,
                 lvc_block_nums=3,
                 lvc_layers_each_block=10,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 dropout=0.0,)
    noise=torch.randn(16,1,8192)
    mel = torch.randn(16, 80, 16)
    x=initmd(noise,mel)
    pass