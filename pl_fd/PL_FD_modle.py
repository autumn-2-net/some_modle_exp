import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F

from math import sqrt
import numpy as np
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffwave import tfff

# tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
from fd.modules import DiffusionDBlock, TimeAware_LVCBlock
Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

writer = SummaryWriter("./mdsr_1000s/", )
def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)
@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)

class FastDiff(nn.Module):
    """FastDiff module."""

    def __init__(self,
                 audio_channels=1,
                 inner_channels=64,
                 cond_channels=128,#mel
                 upsample_ratios=[8, 8, 4,2],
                 lvc_layers_each_block=4,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=128,
                 kpnet_conv_size=3,
                 dropout=0.0,
                 maxstep=1000,
                 diffusion_step_embed_dim_in=128,
                 diffusion_step_embed_dim_mid=512,
                 diffusion_step_embed_dim_out=512,
                 use_weight_norm=True):
        super().__init__()

        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.audio_channels = audio_channels
        self.cond_channels = cond_channels
        self.lvc_block_nums = len(upsample_ratios)
        self.first_audio_conv = nn.Conv1d(1, inner_channels,
                                    kernel_size=7, padding=(7 - 1) // 2,
                                    dilation=1, bias=True)

        # define residual blocks
        self.lvc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        # the layer-specific fc for noise scale embedding
        self.fc_t = nn.ModuleList()
        # self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        # self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        self.defe=DiffusionEmbedding(maxstep)

        cond_hop_length = 1
        for n in range(self.lvc_block_nums):
            cond_hop_length = cond_hop_length * upsample_ratios[n]
            lvcb = TimeAware_LVCBlock(
                in_channels=inner_channels,
                cond_channels=cond_channels,
                upsample_ratio=upsample_ratios[n],
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
                noise_scale_embed_dim_out=diffusion_step_embed_dim_out
            )
            self.lvc_blocks += [lvcb]
            self.downsample.append(DiffusionDBlock(inner_channels, inner_channels, upsample_ratios[self.lvc_block_nums-n-1]))


        # define output layers
        self.final_conv = nn.Sequential(nn.Conv1d(inner_channels, audio_channels, kernel_size=7, padding=(7 - 1) // 2,
                                        dilation=1, bias=True))

        # apply weight norm
        # if use_weight_norm:
        #     self.apply_weight_norm()

    def forward(self, data):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        audio, c, diffusion_steps = data

        # embed diffusion step t
        diffusion_step_embed = self.defe(diffusion_steps)
        # diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        # diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        audio = self.first_audio_conv(audio.unsqueeze(1))
        downsample = []
        for down_layer in self.downsample:
            downsample.append(audio)
            audio = down_layer(audio)

        x = audio
        for n, audio_down in enumerate(reversed(downsample)):
            x = self.lvc_blocks[n]((x, audio_down, c, diffusion_step_embed))

        # apply final layers
        x = self.final_conv(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                # logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

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

class lossfn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn1=nn.MSELoss()
        self.fn2 = nn.L1Loss()
    def forward(self,x,y):
        return (self.fn1(x,y)+self.fn2(x,y))/2



class PL_diffwav(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.diffwav = FastDiff(cond_channels=128)

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
        self.noise_level = noise_level
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = lossfn()
        self.summary_writer = None
        self.grad_norm = 0
        self.lrc = self.params.learning_rate
        self.val_loss = 0
        self.valc = []

    def forward(self, audio, diffusion_step, spectrogram=None):
        a=(audio,spectrogram,diffusion_step)

        return self.diffwav(a)

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
            'spectrogram': batch[1]
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

        predicted = self.forward(noisy_audio, t, spectrogram)
        loss = self.loss_fn(noise, predicted.squeeze(1))
        if self.is_master:
            if self.global_step % 50 == 0:
                if self.global_step!=0:

                    self._write_summary(self.global_step, accc, loss)

        return loss

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

    def on_after_backward(self):
        self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm or 1e9)

    # train

    def _write_summary(self, step, features, loss):  # log 器
        tensorboard = self.logger.experiment
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
        if not self.params.unconditional:
            mel = self.plot_mel([

               features['spectrogram'][:1].detach().cpu().numpy()[0],
            ],
                [ "Ground-Truth Spectrogram"],
            )
            # writer.add_figure('val_' + str(self.global_step) + '/spectrogram', mel, idx)
            # writer.add_figure('feature/spectrogram',mel , step)
        writer.add_scalar('train/loss', loss, step)
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
        aaac, opo = self.predict(spectrogram, fast_sampling=True)
        loss = self.loss_fn(aaac, audio)
        accc['gad'] = aaac
        # print(loss)
        self.val_loss = (loss + self.val_loss) / 2

        accc['spectrogramg'] = tfff.transform(aaac.detach().cpu())
        self.valc.append(accc)

        return loss

    def predict(self, spectrogram=None, fast_sampling=False):
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
    from fd.params import params

    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # torch.backends.cudnn.benchmark = True
    md = PL_diffwav(params)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"lagegeFDbignet_1000")
    dataset = from_path([#'./testwav/',
                         r'K:\dataa\OpenSinger',r'C:\Users\autumn\Desktop\poject_all\DiffSinger\data\raw\opencpop\segments\wavs'], params)
    # dataset= from_path(['./test/', ], params, ifv=True)
    datasetv = from_path(['./test/', ], params, ifv=True)
    md = md.load_from_checkpoint('./mdscp/sample-mnist-epoch19-19-92977.ckpt', params=params)
    # md = torch.compile(md)
    checkpoint_callback = ModelCheckpoint(

    # monitor = 'val/loss',

    dirpath = './mdscp',

    filename = 'sample-mnist-epoch{epoch:02d}-{epoch}-{step}',

    auto_insert_metric_name = False,every_n_epochs=20,save_top_k = -1

    )
    trainer = pl.Trainer(max_epochs=950, logger=tensorboard, devices=-1, benchmark=True, num_sanity_val_steps=1,
                         val_check_interval=2000,callbacks=[checkpoint_callback],
                         #precision=16
                          #resume_from_checkpoint='./bignet/default/version_25/checkpoints/epoch=134-step=1074397.ckpt'
                         )
    trainer.fit(model=md, train_dataloaders=dataset, val_dataloaders=datasetv, )

