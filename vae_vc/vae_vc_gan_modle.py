import itertools
import json
import os

from complexdataset import mel_spectrogram, ComplexDataset

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
from models import Encoder, Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,\
    feature_loss, generator_loss, discriminator_loss
from stft import TorchSTFT
# tensorboard = pl_loggers.TensorBoardLogger(save_dir="")



class PL_diffwav(pl.LightningModule):
    def __init__(self, params,h):
        super().__init__()
        self.params = params
        self.encoder = Encoder(h)
        self.generator = Generator(h)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        # self.model_dir = model_dir
        # self.model = model
        # self.dataset = dataset
        # self.optimizer = optimizer
        # self.params = params
        # self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        # self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True


        self.loss_fn = nn.L1Loss()
        self.summary_writer = None
        self.grad_norm = 0
        self.lrc = self.params.learning_rate
        self.val_loss = 0
        self.valc = []
        # self.D=JCUDiscriminator(len(self.params.noise_schedule))

        self.automatic_optimization = False
        self.h=h

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
        x, y, _, y_mel = batch
        y = y.unsqueeze(1)


        l = self.encoder(x)

        y_g_hat = self.generator(l)

        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), self.h.n_fft, self.h.num_mels, self.h.sampling_rate, self.h.hop_size, self.h.win_size,
                                      self.h.fmin, self.h.fmax_for_loss)




        ####################
        #T D
        ####################

        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f


        ################################
        # D优化
        ####################################
        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)
        opt_d.step()
        ############################

        ##############

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
        # L2 Time-Domain Loss
        loss_wav = F.mse_loss(y, y_g_hat) * 100
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_wav

        loss=F.l1_loss(y,y_g_hat)



        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        opt_g.step()

        if self.is_master:
            if self.global_step % 10 == 0 or (self.global_step-1) % 10 == 0:
                fr={'spectrogram':y_mel,'audio':y}

                self._write_summary(self.global_step, fr, loss,loss_disc_all=loss_disc_all,loss_disc_s=loss_disc_s,loss_disc_f=loss_disc_f,loss_mel=loss_mel
                                    ,loss_wav=loss_wav,
                                    loss_gen_all=loss_gen_all
                                        )



    def configure_optimizers(self):

        opt_g = torch.optim.AdamW(itertools.chain(self.encoder.parameters(), self.generator.parameters()), lr=self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])
        opt_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()), lr=self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])
        lrg=torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.h.lr_decay, )
        lrd=torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.h.lr_decay, )
        return [opt_g, opt_d], [lrg,lrd]
        # return {"optimizer": optimizer,
        #         # "lr_scheduler": lt
        #         }

    def on_train_epoch_end(self):
        lrg,lrd = self.lr_schedulers()
        lrg.step()
        lrd.step()

    def on_after_backward(self):
        self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm or 1e9)

    # train

    def _write_summary(self, step, features, loss,loss_disc_all,loss_disc_s,loss_disc_f,loss_gen_all,loss_wav,loss_mel):  # log 器
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
        writer.add_scalar('train/loss_disc_all', loss_disc_all, step)
        writer.add_scalar('train/loss_disc_s', loss_disc_s, step)
        writer.add_scalar('train/loss_disc_f', loss_disc_f, step)

        writer.add_scalar('train/loss_gen_all', loss_gen_all, step)
        writer.add_scalar('train/loss_wav', loss_wav, step)
        writer.add_scalar('train/loss_mel', loss_mel, step)



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

        x, y, _, y_mel = batch
        # self.valc=accc
        l = self.encoder(x)
        y_g_hat = self.generator(l)


        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), self.h.n_fft, self.h.num_mels, self.h.sampling_rate,
                                      self.h.hop_size, self.h.win_size,
                                      self.h.fmin, self.h.fmax_for_loss)
        loss=F.l1_loss(y_mel, y_g_hat_mel)
        self.val_loss = (self.val_loss+loss)/2
        self.valc.append({'spectrogram':y_g_hat_mel,'spectrogramg':y_mel,'audio':y,'gad':y_g_hat})



        return loss



if __name__ == "__main__":
    from diffwave.dataset2 import from_path, from_gtzan
    from diffwave.params import params

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # torch.backends.cudnn.benchmark = True
    with open('./diffwave/config.json') as f:
        data = f.read()


    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self


    json_config = json.loads(data)
    h = AttrDict(json_config)
    md = PL_diffwav(params,h)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="bignet_1000")
    tds=ComplexDataset([#'./testwav/',
                         r'K:\dataa\OpenSinger',r'C:\Users\autumn\Desktop\poject_all\DiffSinger\data\raw\opencpop\segments\wavs'], h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, True, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss,
                             )
    dataset = from_path(tds, params)
    tdsv = ComplexDataset(['./test/', ],
        h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss,
    )
    datasetv = from_path(tdsv, params, ifv=True)

    md = md.load_from_checkpoint(r'C:\Users\autumn\Desktop\poject_all\vcoder\vae_vc\bignet_1000\lightning_logs\version_0\checkpoints\epoch=9-step=95654.ckpt', params=params,h=h)

    # eee=torch.load('a.cpt')
    # md.load_state_dict(eee['state_dict'])
    # ccc=torch.load('./bignet_1000/lightning_logs/version_5/checkpoints/epoch=21-step=205675.ckpt')['state_dict']
    # aca=torch.load(r'C:\Users\autumn\Desktop\poject_all\vcoder\bignet\default\version_27\checkpoints\epoch=190-step=1315282.ckpt')['state_dict']


    # for i in ccc:
    #     w=aca.get(i)
    #     if w is not None:
    #         ccc[i]=w
    #         # print(w)
    #     else:
    #         ccc[i] = torch.randn_like(ccc[i])
    #         torch.randn_like(ccc[i])
    # dddd=torch.load('./bignet_1000/lightning_logs/version_5/checkpoints/epoch=21-step=205675.ckpt')
    # dddd['state_dict']=ccc
    # torch.save(dddd,'a.cpt')
    # for i in dataset:
    #     print(i)

    trainer = pl.Trainer(max_epochs=250, logger=tensorboard, devices=-1, benchmark=True, num_sanity_val_steps=3,
                         val_check_interval=params.valst,
                         #precision=16
                          #precision='bf16'
                          #resume_from_checkpoint='./bignet/default/version_25/checkpoints/epoch=134-step=1074397.ckpt'
                         )
    trainer.fit(model=md, train_dataloaders=dataset, val_dataloaders=datasetv, )

