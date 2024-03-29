import os
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F

from math import sqrt
import numpy as np
from pytorch_lightning import loggers as pl_loggers
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffwave import tfff

# tensorboard = pl_loggers.TensorBoardLogger(save_dir="")

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):  # 这里有点坑 这里是mel的上采样
    def __init__(self, n_mels):
        super().__init__()
        if n_mels == 256:
            self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
            self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        if n_mels == 512:
            self.conv1 = ConvTranspose2d(1, 1, [3, 64], stride=[1, 32], padding=[1, 16])
            self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):  # 残差块吧
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond:  # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)  # ??????????????
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            xcxc = self.dilated_conv(y)
            y = xcxc + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal

class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, T ):
        super(JCUDiscriminator, self).__init__()

        n_mel_channels = 1
        residual_channels = T
        n_layer = 3
        n_uncond_layer = 2
        n_cond_layer = 2
        n_channels = [64,128,512,128,1]
        kernel_sizes = [128,32,15,5,3]
        strides = [32,4,6,1,1]
        p=[65,33,14,0,0]


        self.input_projection = LinearNorm(2 * n_mel_channels, 2 * n_mel_channels)
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        # self.mlp = nn.Sequential(
        #     LinearNorm(residual_channels, residual_channels * 4),
        #     Mish(),
        #     LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        # )

        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,padding=p[i]
                )
                for i in range(n_layer)
            ]
        )
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,padding=p[i]
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,padding=p[i]
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, t):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x_t_prevs=x_t_prevs.transpose(1, 2)
        x_ts=x_ts.transpose(1, 2)
        x = self.input_projection(
            torch.cat([x_t_prevs, x_ts], dim=-1)
        ).transpose(1, 2)
        diffusion_step = self.diffusion_embedding(t).unsqueeze(-1)


        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        x_cond = (x + diffusion_step)
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats
class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        if self.params.unconditional:  # use unconditional model  #不知道干什么的
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.hop_samples)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
class DResidualBlock(nn.Module):  # 残差块吧
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)


        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):
        # assert (conditioner is None and self.conditioner_projection is None) or \
        #        (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        if conditioner:
            y = x + diffusion_step
        else:
            y=x

        y = self.dilated_conv(y)


        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
class WavenetDiscriminator(nn.Module):
    def __init__(self,T_len,residual_channels,lay=10,ducycle=10):
        super().__init__()
        self.residual_layers_con = nn.ModuleList([
            DResidualBlock(0, residual_channels, 2 ** (i % ducycle),
                          )
            for i in range(lay)
        ])
        self.residual_layers_uncon = nn.ModuleList([
            DResidualBlock(0, residual_channels, 2 ** (i % ducycle),
                           )
            for i in range(lay)
        ])
        self.diffusion_embedding = DiffusionEmbedding(T_len)
        self.input_projection = Conv1d(2, residual_channels, 1)
        self.skip_projection_con = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection_con = Conv1d(residual_channels, 1, 1)
        self.skip_projection_uncon = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection_uncon = Conv1d(residual_channels, 1, 1)
    def forward(self, x_ts, x_t_prevs, t):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x_t_prevs=x_t_prevs.transpose(1, 2)
        x_ts=x_ts.transpose(1, 2)
        x = self.input_projection(
            torch.cat([x_t_prevs, x_ts], dim=-1).transpose(1, 2)
        )
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(t)

        skip = None
        x_con=x
        for layer in self.residual_layers_con:
            x_con, skip_connection = layer(x_con, diffusion_step, True)
            skip = skip_connection if skip is None else skip_connection + skip

        x_con = skip / sqrt(len(self.residual_layers_con))
        x_con = self.skip_projection_con(x_con)
        x_con = F.relu(x_con)
        x_con = self.output_projection_con(x_con)

        skip = None
        x_uncon = x
        for layer in self.residual_layers_uncon:
            x_uncon, skip_connection = layer(x_uncon, diffusion_step, False)
            skip = skip_connection if skip is None else skip_connection + skip

        x_uncon = skip / sqrt(len(self.residual_layers_uncon))
        x_uncon = self.skip_projection_con(x_uncon)
        x_uncon = F.relu(x_uncon)
        x_uncon = self.output_projection_con(x_uncon)
        return [x_con],[x_uncon]


        # cond_feats = []
        # uncond_feats = []
        # for layer in self.conv_block:
        #     x = F.leaky_relu(layer(x), 0.2)
        #     cond_feats.append(x)
        #     uncond_feats.append(x)
        #
        # x_cond = (x + diffusion_step)
        # x_uncond = x
        #
        # for layer in self.cond_conv_block:
        #     x_cond = F.leaky_relu(layer(x_cond), 0.2)
        #     cond_feats.append(x_cond)
        #
        # for layer in self.uncond_conv_block:
        #     x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
        #     uncond_feats.append(x_uncond)
        # return cond_feats, uncond_feats

class PL_diffwav(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.diffwav = DiffWave(self.params)

        # self.model_dir = model_dir
        # self.model = model
        # self.dataset = dataset
        # self.optimizer = optimizer
        # self.params = params
        # self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        # self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        noise_level = torch.tensor(noise_level.astype(np.float32))
        self.alpha = torch.tensor((1 - beta).astype(np.float32))
        self.beta=torch.tensor(beta.astype(np.float32))
        self.noise_level = noise_level
        self.loss_fn = nn.L1Loss()
        self.summary_writer = None
        self.grad_norm = 0
        self.lrc = self.params.learning_rate
        self.val_loss = 0
        self.valc = []
        # self.D=JCUDiscriminator(len(self.params.noise_schedule))
        self.D = WavenetDiscriminator(len(self.params.noise_schedule),32)
        self.automatic_optimization = False

    def forward(self, audio, diffusion_step, spectrogram=None):

        return self.diffwav(audio, diffusion_step, spectrogram)

    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def training_step(self, batch, batch_idx,):
        # training_step defines the train loop.
        # it is independent of forward
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        # return loss
        # self.step = self.step+1
        pass
        opt_g, opt_d=self.optimizers()
        accc = {
            'audio': batch[0],
            'spectrogram': batch[1]
        }

        audio = accc['audio']
        spectrogram = accc['spectrogram']

        N, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale ** 0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise



        ####################
        #T D
        ####################
        nns=[]
        noise_scale1 = self.noise_level[t - 1].unsqueeze(1)
        noise_scale_sqrt1 = noise_scale ** 0.5
        for idx,i in enumerate(t):
            if i>0:
                # ccc=audio[idx]
                na_1 = noise_scale_sqrt1[idx] * audio[idx] + (1.0 - noise_scale1[idx]) ** 0.5 * noise[idx]
                nns.append(na_1.unsqueeze(0))
            else:
                na_1=audio[idx]
                nns.append(na_1.unsqueeze(0))
        na_1=torch.cat(nns, dim=0)

        predicted = self.forward(noisy_audio, t, spectrogram)
        loss = self.loss_fn(noise, predicted.squeeze(1))
        nnc=[]
        for idx, i in enumerate(t):
            c1 = 1 / self.alpha[i] ** 0.5
            c2 = self.beta[i] / (1 - self.noise_level[i]) ** 0.5
            # aas=predicted.squeeze(1)
            na = c1 * (noisy_audio[idx] - c2 *( predicted.squeeze(1)[idx]))
            if i > 0:
                noise = torch.randn_like(audio[idx])
                sigma = ((1.0 - self.noise_level[i - 1]) / (1.0 - self.noise_level[i]) * self.beta[i]) ** 0.5
                na += sigma * noise
                nnc.append(na.unsqueeze(0))



            else:
                # na_1 = na
                nnc.append(na.unsqueeze(0))
        Gna_1 = torch.cat(nnc, dim=0)
        if self.global_step>1:
            cond_featsF, uncond_featsF = self.D(noisy_audio.unsqueeze(1), Gna_1.unsqueeze(1).detach(), t)
            cond_featsT, uncond_featsT = self.D(noisy_audio.unsqueeze(1), na_1.unsqueeze(1).detach(), t)



            F_loss_con=F.mse_loss(cond_featsF[-1], torch.zeros_like(cond_featsF[-1]))
            F_loss_uncon = F.mse_loss(uncond_featsF[-1], torch.zeros_like(uncond_featsF[-1]))

            T_loss_con = F.mse_loss(cond_featsT[-1], torch.ones_like(cond_featsT[-1]))
            T_loss_uncon = F.mse_loss(uncond_featsT[-1], torch.ones_like(uncond_featsT[-1]))

            D_mix_loss=(F_loss_con+F_loss_uncon+T_loss_con+T_loss_uncon)/4
            ################################
            # D优化
            ####################################
            opt_d.zero_grad()
            self.manual_backward(D_mix_loss)
            opt_d.step()
        ############################
        else:
            F_loss_con=0
            F_loss_uncon=0
            T_loss_con=0
            T_loss_uncon=0
        #G Train
        ##############
        if self.global_step > 1000:
            Gcond_featsF, Guncond_featsF = self.D(noisy_audio.unsqueeze(1), Gna_1.unsqueeze(1), t)
            G_loss_con = F.mse_loss(Gcond_featsF[-1], torch.ones_like(Gcond_featsF[-1]))
            G_loss_uncon = F.mse_loss(Guncond_featsF[-1], torch.ones_like(Guncond_featsF[-1]))
        else:
            G_loss_con=0
            G_loss_uncon=0

        G_loss=(loss+(G_loss_con+G_loss_uncon)/2)


        opt_g.zero_grad()
        self.manual_backward(G_loss)
        opt_g.step()

        if self.is_master:
            if self.global_step % 10 == 0 or (self.global_step-1) % 10 == 0:

                self._write_summary(self.global_step, accc, loss,DTloss_unc=T_loss_uncon,DFloss_unc=F_loss_uncon,DFloss_c=F_loss_con,DTloss_c=T_loss_con,
                                        Gdloss_c=G_loss_con,Gdloss_unc=G_loss_uncon
                                        )



        # return loss
        # if optimizer_idx == 0:
        #
        #
        #     predicted = self.forward(noisy_audio, t, spectrogram)
        #     loss = self.loss_fn(noise, predicted.squeeze(1))
        #     c1 = 1 / self.alpha[t] ** 0.5
        #     c2 = self.beta[t] / (1 - self.noise_level[t]) ** 0.5
        #     na = c1 * (noisy_audio - c2 * predicted.squeeze(1))
        #     if t > 0:
        #         noise = torch.randn_like(audio)
        #         sigma = ((1.0 - self.noise_level[t - 1]) / (1.0 - self.noise_level[t]) * self.beta[t]) ** 0.5
        #         na += sigma * noise
        #     cond_feats, uncond_feats=  self.D(noisy_audio,na,t)
        #
        #     if self.is_master:
        #         if self.global_step % 50 == 0:
        #             if self.global_step != 0:
        #                 self._write_summary(self.global_step, accc, loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        lt = {
            "scheduler": MultiStepLR(optimizer, self.params.lrcl, self.params.lrcc),  # 调度器
            "interval": self.params.interval,  # 调度的单位，epoch或step
            "frequency": self.params.frequency,  # 调度的频率，多少轮一次
            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }
        opt_g = torch.optim.AdamW(self.diffwav.parameters(), lr=self.params.learning_rate)
        opt_d = torch.optim.AdamW(self.D.parameters(), lr=self.params.learning_rate)
        return [opt_g, opt_d], []
        # return {"optimizer": optimizer,
        #         # "lr_scheduler": lt
        #         }

    def on_after_backward(self):
        self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm or 1e9)

    # train

    def _write_summary(self, step, features, loss,DTloss_unc,DTloss_c,DFloss_unc,DFloss_c,Gdloss_unc,Gdloss_c):  # log 器
        # tensorboard = self.logger.experiment
        # writer = tensorboard.SummaryWriter
        writer = SummaryWriter("./mdsr_1000/", purge_step=step)
        # writer = tensorboard
        writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
        if not self.params.unconditional:
            mel = self.plot_mel([

               features['spectrogram'][:1].detach().cpu().numpy()[0],
            ],
                [ "Ground-Truth Spectrogram"],
            )
            # writer.add_figure('val_' + str(self.global_step) + '/spectrogram', mel, idx)
            writer.add_figure('feature/spectrogram',mel , step)
        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/DTloss_unc', DTloss_unc, step)
        writer.add_scalar('train/DTloss_c', DTloss_c, step)
        writer.add_scalar('train/DFloss_unc', DFloss_unc, step)
        writer.add_scalar('train/DFloss_c', DFloss_c, step)
        writer.add_scalar('train/Gdloss_unc', Gdloss_unc, step)
        writer.add_scalar('train/Gdloss_c', Gdloss_c, step)

        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.add_scalar('train/lr', self.lrc, step)
        writer.flush()
        self.summary_writer = writer

    def mmmmd(self, t):

        # axe.set_xticks(np.arange(len(x_labels)))
        # axe.set_yticks(np.arange(len(y_labels)))
        # axe.set_xticklabels(x_labels)
        # axe.set_yticklabels(y_labels   )

        fig, axe = plt.subplots(figsize=(20, 10))
        im = axe.imshow(t.detach().cpu().numpy())
        # plt.show()
        return im

    def plot_mel(self, data, titles=None):
        fig, axes = plt.subplots(len(data), 1, squeeze=False,figsize = (15, 10))
        if titles is None:
            titles = [None for i in range(len(data))]
        plt.tight_layout()

        for i in range(len(data)):
            mel = data[i]
            if isinstance(mel, torch.Tensor):
                mel = mel.detach().cpu().numpy()
            axes[i][0].imshow(mel, origin="lower")
            axes[i][0].set_aspect(2.5, adjustable="box")
            axes[i][0].set_ylim(0, mel.shape[0])
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
            axes[i][0].set_anchor("W")

        return fig

    def on_validation_end(self):
        writer = SummaryWriter("./mdsr_1000/", purge_step=self.global_step)
        writer.add_scalar('val/loss', self.val_loss, self.global_step)

        for idx, i in enumerate(self.valc):
            writer.add_audio('val_' + str(self.global_step) + '/audio_gt', i['audio'][0], idx,
                             sample_rate=self.params.sample_rate)
            writer.add_audio('val_' + str(self.global_step) + '/audio_g', i['gad'][0], idx,
                             sample_rate=self.params.sample_rate)
            # writer.add_figure('val_'+str(self.global_step)+'/GT_spectrogram', self.mmmmd(torch.flip(i['spectrogram'][:1], [1])), idx)
            # writer.add_figure('val_'+str(self.global_step)+'/G_spectrogram', self.mmmmd(torch.flip(i['spectrogramg'][:1], [1])), idx)
            mel = self.plot_mel([
               i['spectrogram'][:1].detach().cpu().numpy()[0],
                i['spectrogramg'][:1].detach().cpu().numpy()[0],
            ],
                ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
            )
            writer.add_figure('val_' + str(self.global_step) + '/spectrogram', mel, idx)

    def validation_step(self, batch, idx):
        # print(idx)
        if idx == 0:
            self.val_loss = 0
            self.valc = []

        accc = {
            'audio': batch[0],
            'spectrogram': batch[1]
        }
        # self.valc=accc

        audio = accc['audio']
        spectrogram = accc['spectrogram']
        aaac, opo = self.predict(spectrogram)
        loss = self.loss_fn(aaac, audio)
        accc['gad'] = aaac
        # print(loss)
        self.val_loss = (loss + self.val_loss) / 2

        accc['spectrogramg'] = tfff.transform(aaac.detach().cpu())
        self.valc.append(accc)

        return loss

    def predict(self, spectrogram=None, fast_sampling=True):
        # Lazy load model.
        device = spectrogram.device

        with torch.no_grad():
            # Change in notation from the DiffWave paper for fast sampling.
            # DiffWave paper -> Implementation below
            # --------------------------------------
            # alpha -> talpha
            # beta -> training_noise_schedule
            # gamma -> alpha
            # eta -> beta
            training_noise_schedule = np.array(self.params.noise_schedule)
            inference_noise_schedule = np.array(
                self.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                                talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            if not self.params.unconditional:
                if len(spectrogram.shape) == 2:  # Expand rank 2 tensors by adding a batch dimension.
                    spectrogram = spectrogram.unsqueeze(0)
                spectrogram = spectrogram.to(device)
                audio = torch.randn(spectrogram.shape[0], self.params.hop_samples * spectrogram.shape[-1],
                                    device=device)
            else:
                audio = torch.randn(1, self.paramsaudio_len, device=device)

            for n in tqdm(range(len(alpha) - 1, -1, -1)):
                # print(n)
                # for n in range(len(alpha) - 1, -1, -1):  #扩散过程
                c1 = 1 / alpha[n] ** 0.5
                c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
                audio = c1 * (audio - c2 * self.forward(audio, torch.tensor([T[n]], device=audio.device),
                                                        spectrogram).squeeze(
                    1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                    audio += sigma * noise
                # audio = torch.clamp(audio, -1.0, 1.0)
        return audio, self.params.sample_rate


if __name__ == "__main__":
    from diffwave.dataset2 import from_path, from_gtzan
    from diffwave.params import params

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # torch.backends.cudnn.benchmark = True
    md = PL_diffwav(params)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="bignet_1000")
    dataset = from_path([#'./testwav/',
                         r'K:\dataa\OpenSinger',r'C:\Users\autumn\Desktop\poject_all\DiffSinger\data\raw\opencpop\segments\wavs'], params)
    datasetv = from_path(['./test/', ], params, ifv=True)
    md = md.load_from_checkpoint('a.cpt', params=params)
    # eee=torch.load('a.cpt')
    # md.load_state_dict(eee['state_dict'])
    ccc=torch.load('./bignet_1000/lightning_logs/version_5/checkpoints/epoch=21-step=205675.ckpt')['state_dict']
    aca=torch.load(r'C:\Users\autumn\Desktop\poject_all\vcoder\bignet\default\version_27\checkpoints\epoch=190-step=1315282.ckpt')['state_dict']
    for i in ccc:
        w=aca.get(i)
        if w is not None:
            ccc[i]=w
            # print(w)
        else:
            ccc[i] = torch.randn_like(ccc[i])
            torch.randn_like(ccc[i])
    dddd=torch.load('./bignet_1000/lightning_logs/version_5/checkpoints/epoch=21-step=205675.ckpt')
    dddd['state_dict']=ccc
    torch.save(dddd,'a.cpt')

    trainer = pl.Trainer(max_epochs=250, logger=tensorboard, devices=-1, benchmark=True, num_sanity_val_steps=1,
                         val_check_interval=params.valst,
                         precision=16
                          #precision='bf16'
                          #resume_from_checkpoint='./bignet/default/version_25/checkpoints/epoch=134-step=1074397.ckpt'
                         )
    trainer.fit(model=md, train_dataloaders=dataset, val_dataloaders=datasetv, )

