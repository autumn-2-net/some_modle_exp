import os

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

from logger import utils
#from data_loaders import get_data_loaders
#from solver import train
from ddsp.vocoder import Sins, CombSub
from ddsp.loss import HybridLoss



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
class ddsp_adp(nn.Module):
    def __init__(self,args):
        super().__init__()

        if args.model.type == 'Sins':
            self.ddsp_model = Sins(
                sampling_rate=args.data.sampling_rate,
                block_size=args.data.block_size,
                n_harmonics=args.model.n_harmonics,
                n_mag_allpass=args.model.n_mag_allpass,
                n_mag_noise=args.model.n_mag_noise,
                n_mels=args.data.n_mels)

        elif args.model.type == 'CombSub':
            self.ddsp_model = CombSub(
                sampling_rate=args.data.sampling_rate,
                block_size=args.data.block_size,
                n_mag_allpass=args.model.n_mag_allpass,
                n_mag_harmonic=args.model.n_mag_harmonic,
                n_mag_noise=args.model.n_mag_noise,
                n_mels=args.data.n_mels)

        else:
            raise ValueError(f" [x] Unknown Model: {args.model.type}")
        self.args=args
        self.loss=HybridLoss(args.data.block_size, args.loss.fft_min, args.loss.fft_max, args.loss.n_scale, args.loss.lambda_uv, args.device)
    def forward(self,mel,f0:torch.tensor,):
        signal, _, (s_h, s_n) = self.ddsp_model(mel.transpose(1,2), f0.unsqueeze(2), max_upsample_dim=self.args.train.max_upsample_dim)

        # loss

        return signal,s_h


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
            # self.conv1 = ConvTranspose2d(1, 1, [3, 64], stride=[1, 32], padding=[1, 16])
            # self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
            self.conv1 = ConvTranspose2d(1, 5, [3, 64], stride=[1, 32], padding=[1, 16])
            self.conv2 = ConvTranspose2d(5, 1, [3, 32], stride=[1, 16], padding=[1, 8])#
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
        # self.spectrogram_upsampler = SpectrogramUpsampler(params.hop_samples)
        if not uncond:  # conditional model
            # self.conditioner_projection = Conv1d(1, 2 * residual_channels, 257,padding=128)  # ?????????????? 傻了改错了
            self.conditioner_mel = Conv1d(n_mels, 2 * residual_channels, 1, )
            # todo
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step,mel,):
        # assert (conditioner is None and self.conditioner_projection is None) or \
        #        (conditioner is not None and self.conditioner_projection is not None)
        # mel = self.spectrogram_upsampler(mel)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        conditioner =  self.conditioner_mel(mel)
        xcxc = self.dilated_conv(y)
        y = xcxc + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(2, params.residual_channels, 1)

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


    def forward(self, audio, diffusion_step, mel,spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x=torch.cat([x,spectrogram],dim=1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            mel = self.spectrogram_upsampler(mel)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, mel)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class PL_diffwav(pl.LightningModule):
    def __init__(self, params,argss):
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
        self.ddsp=ddsp_adp(args=argss)
        self.args = argss
        args=argss

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        noise_level = torch.tensor(noise_level.astype(np.float32))
        self.noise_level = noise_level
        self.loss_fn = nn.L1Loss()
        self.summary_writer = None
        self.grad_norm = 0
        self.lrc = self.params.learning_rate
        self.val_loss = 0
        self.valc = []
        self.loss = HybridLoss(args.data.block_size, args.loss.fft_min, args.loss.fft_max, args.loss.n_scale,
                               args.loss.lambda_uv, args.device)

    def forward(self, audio, diffusion_step,f0, spectrogram=None):
        wavv,s_h=self.ddsp(spectrogram,f0,)
        wavv=wavv.type_as(audio)
        return self.diffwav(audio, diffusion_step,spectrogram, wavv.unsqueeze(
                    1)),wavv,s_h

    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def training_step(self, batch, batch_idx):
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
        accc = {
            'audio': batch[0],
            'spectrogram': batch[1],'f0': batch[2],'uv': batch[3],
        }

        audio = accc['audio']
        spectrogram = accc['spectrogram']

        N, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale ** 0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

        predicted,locc,s_h = self.forward(noisy_audio, t,f0=accc['f0'] ,spectrogram=spectrogram,)
        detach_uv = False
        if self.global_step < self.args.loss.detach_uv_step:
            detach_uv = True
        losscc, (loss_rss, loss_uv) = self.loss(locc, s_h, audio, accc['uv'], detach_uv=detach_uv,
                                              uv_tolerance=self.args.loss.uv_tolerance)

        # handle nan loss
        if torch.isnan(losscc):
            raise ValueError(' [x] nan loss ')
        loss = self.loss_fn(noise, predicted.squeeze(1))
        mix_loss=(loss*4+losscc)/5
       # mix_loss=loss
        if self.is_master:
            if self.global_step % 10 == 0:
                if self.global_step!=0:

                    self._write_summary(self.global_step, accc, loss,mix_loss,losscc)

        return mix_loss

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
        return {"optimizer": optimizer,
                # "lr_scheduler": lt
                }

    # def on_after_backward(self):
    #     self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm or 1e9)

    # train

    def _write_summary(self, step, features, loss,mix_loss,losscc):  # log 器
        tensorboard = self.logger.experiment
        # writer = tensorboard.SummaryWriter
        # writer = SummaryWriter("./mdsr/", purge_step=step)
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
        writer.add_scalar('train/mix_loss', mix_loss, step)
        writer.add_scalar('train/ddsp_loss', losscc, step)
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
        fig, axes = plt.subplots(len(data), 1, squeeze=False,figsize = (15,10 ))
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

        writer.add_scalar('val/loss', self.val_loss, self.global_step)

        for idx, i in enumerate(self.valc):
            writer.add_audio('val_' + str(self.global_step) + '/audio_gt', i['audio'][0], idx,
                             sample_rate=self.params.sample_rate)
            writer.add_audio('val_' + str(self.global_step) + '/audio_g', i['gad'][0], idx,
                             sample_rate=self.params.sample_rate)
            writer.add_audio('val_' + str(self.global_step) + '/audio_cddsp', i['ddspw'][0], idx,
                             sample_rate=self.params.sample_rate)

            # writer.add_figure('val_'+str(self.global_step)+'/GT_spectrogram', self.mmmmd(torch.flip(i['spectrogram'][:1], [1])), idx)
            # writer.add_figure('val_'+str(self.global_step)+'/G_spectrogram', self.mmmmd(torch.flip(i['spectrogramg'][:1], [1])), idx)
            mel = self.plot_mel([
               i['spectrogram'][:1].detach().cpu().numpy()[0],
                i['spectrogramg'][:1].detach().cpu().numpy()[0],

            ],
                ["Sampled Spectrogram", "Ground-Truth Spectrogram",],
            )
            writer.add_figure('val_' + str(self.global_step) + '/spectrogram', mel, idx)
            mels = self.plot_mel([
                i['ddsps'][:1].detach().cpu().numpy()[0],
                i['spectrogramg'][:1].detach().cpu().numpy()[0],

            ],
                ["ddsp", "Ground-Truth Spectrogram", ],
            )

            writer.add_figure('val_' + str(self.global_step) + '/spectrogram_ddsp_', mels, idx)
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
        uv =batch[3]
        f0 = batch[2]
        aaac, opo ,wavv= self.predict(f0,spectrogram,fast_sampling=True)
        loss = self.loss_fn(aaac, audio)

        accc['gad'] = aaac
        accc['ddspw'] = wavv
        # print(loss)
        self.val_loss = (loss + self.val_loss) / 2

        accc['spectrogramg'] = tfff.transform(aaac.detach().cpu())
        accc['ddsps'] = tfff.transform(wavv.detach().cpu())
        self.valc.append(accc)

        return loss

    def predict(self, f0,spectrogram=None, fast_sampling=False):
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
                audio, wavv, s_h=self.forward(audio, torch.tensor([T[n]], device=audio.device),f0,
                                                        spectrogram)

                audio,= c1 * (audio - c2 * audio.squeeze(
                    1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                    audio += sigma * noise
                # audio = torch.clamp(audio, -1.0, 1.0)
        return audio, self.params.sample_rate,wavv


if __name__ == "__main__":
    from diffwave.dataset2 import from_path, from_gtzan
    from diffwave.params import params

    writer = SummaryWriter("./mdsrs/", )

    # torch.backends.cudnn.benchmark = True
    args = utils.load_config('./configs/combsub.yaml')
    md = PL_diffwav(params,argss=args)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="bignet_mix_mel")
    dataset = from_path([#'./testwav/',
                         r'K:\dataa\OpenSinger',#r'C:\Users\autumn\Desktop\poject_all\DiffSinger\data\raw\opencpop\segments\wavs'
    ], params)
    datasetv = from_path(['./test/', ], params, ifv=True)
    #md = md.load_from_checkpoint('./bignet/default/version_13/checkpoints/epoch=6-step=69797.ckpt', params=params)
    trainer = pl.Trainer(max_epochs=250, logger=tensorboard, benchmark=True, num_sanity_val_steps=1,devices=-1,accelerator='gpu',
                         val_check_interval=params.valst,
                        #resume_from_checkpoint='./bignet_mix_mel/default/version_1/checkpoints/epoch=4-step=10611.ckpt'
                         )
    trainer.fit(model=md, train_dataloaders=dataset, val_dataloaders=datasetv, )

