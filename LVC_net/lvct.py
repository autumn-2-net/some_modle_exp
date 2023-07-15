import os
from typing import OrderedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F

from math import sqrt
import numpy as np
from einops import rearrange
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from T_lvc import LVCNetGenerator
from losses import PWGLoss
from lvc import tfff

# tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
# from fd.modules import DiffusionDBlock, TimeAware_LVCBlock
# from fd.sc import V3LSGDRLR

class Upcon(nn.Module):

    def __init__(self):
        super().__init__()
        self.UP = torch.nn.ConvTranspose2d(1, 4, [3, 32], stride=[1, 2], padding=[1, 15])
        self.c1 = torch.nn.Conv2d(4, 4, 3,  padding=1)
        self.c2 = torch.nn.Conv2d(4, 4, 5,  padding=2)
        self.cc2 = torch.nn.Conv2d(4, 4, 7, padding=3)
        self.ccc2 = torch.nn.Conv2d(4, 4, 9, padding=4)
        self.c3 = torch.nn.Conv2d(4, 1,kernel_size=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.UP(x)
        x = F.leaky_relu(x, 0.4)
        x=F.leaky_relu(self.c1(x), 0.4)+x
        x = F.leaky_relu(self.c2(x), 0.4) + x
        x = F.leaky_relu(self.cc2(x), 0.4) + x
        x = F.leaky_relu(self.ccc2(x), 0.4) + x
        x = F.leaky_relu(self.c3(x), 0.4)

        # x = self.conv2(x)
        # x = F.leaky_relu(x, 0.4)
        spectrogram = torch.squeeze(x, 1)
        return spectrogram



class PL_diffwav(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.diffwav = LVCNetGenerator(in_channels=64,
                 out_channels=1,
                 inner_channels=32,
                 cond_channels=128,
                 cond_hop_length=256,
                 lvc_block_nums=4,
                 lvc_layers_each_block=10,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=96,
                 kpnet_conv_size=3,
                 dropout=0.0,)

        # self.model_dir = model_dir
        # self.model = model
        # self.dataset = dataset
        # self.optimizer = optimizer
        # self.params = params
        # self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        # self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True
        self.lossx=PWGLoss(stft_loss_params={ 'fft_sizes':[#2048,
                                                           1024, 2048, 512#,128
                                                           ],
                 'hop_sizes':[#256,
                              120, 240, 50#,25
                              ],
                 'win_lengths':[#1024,
                                600, 1200, 240#,64
                 ],
                 'window':"hann_window"})



        self.UP=Upcon()

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = lossfn()
        self.summary_writer = None
        self.grad_norm = 0
        self.lrc = self.params.learning_rate
        self.val_loss = []
        self.val_loss1 = []
        self.val_loss2 = []
        self.valc = []

    def forward(self, audio, spectrogram):

        spectrogram = self.UP(spectrogram)

        # x = self.conv2(x)
        # x = F.leaky_relu(x, 0.4)



        return self.diffwav(audio,spectrogram)

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


        audio = accc['audio'].unsqueeze(1)
        spectrogram = accc['spectrogram']
        device = audio.device

        b, c, t = spectrogram.size()
        noist = torch.randn(b, 64, t * 512,device=device).type_as(audio)

        aaac = self(noist, spectrogram)
        loss=self.loss_fn(aaac,audio)
        loss1,loxs2=self.lossx.stft_loss(aaac, audio)

        if self.global_step %50==0:
            self._write_summary(self.global_step, accc, loss,aaac,loss1,loxs2)

        loss=loss1+loxs2+loss



        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        # lt = {
        #     "scheduler": V3LSGDRLR(optimizer,),  # 调度器
        #     "interval": 'step',  # 调度的单位，epoch或step
        #     #"frequency": self.params.frequency,  # 调度的频率，多少轮一次
        #     "reduce_on_plateau": False,  # ReduceLROnPlateau
        #     "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
        #     "strict": False  # 如果没有monitor，是否中断训练
        # }
        return {"optimizer": optimizer,
                #"lr_scheduler": lt
                }

    # def on_after_backward(self):
    #     self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm or 1e9)

    # train

    def _write_summary(self, step, features, loss,pre_audio,sc_loss, mag_loss):  # log 器
        # tensorboard = self.logger.experiment
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)

        # if not self.params.unconditional:
        #     mel = self.plot_mel([
        #
        #        features['spectrogram'][:1].detach().cpu().numpy()[0],
        #     ],
        #         [ "Ground-Truth Spectrogram"],

        #     )
        if step %500==0:
            writer.add_audio('T_' + '' + '/audio_P', pre_audio[0], step,
                             sample_rate=self.params.sample_rate)

            # writer.add_figure('val_' + str(self.global_step) + '/spectrogram', mel, idx)
            # writer.add_figure('feature/spectrogram',mel , step)
        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/sc_loss', sc_loss, step)
        writer.add_scalar('train/mag_loss', mag_loss, step)
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
        ppp=0
        exex=len(self.val_loss)
        for ix in self.val_loss:
            ppp=ppp+ix
        lossss=ppp/exex
        ppp = 0
        exex = len(self.val_loss1)
        for ix in self.val_loss1:
            ppp = ppp + ix
        lossss1 = ppp / exex
        ppp = 0
        exex = len(self.val_loss2)
        for ix in self.val_loss2:
            ppp = ppp + ix
        lossss2 = ppp / exex



        writer.add_scalar('val/loss', lossss, self.global_step)
        writer.add_scalar('val/sc_loss', lossss1, self.global_step)
        writer.add_scalar('val/mag_loss', lossss2, self.global_step)

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
            self.val_loss = []
            self.valc = []

        accc = {
            'audio': batch[0],
            'spectrogram': batch[1]
        }

        # self.valc=accc

        audio = accc['audio']
        spectrogram = accc['spectrogram']
        # b,c,t=spectrogram.size()
        device = audio.device

        b, c, t = spectrogram.size()
        noist = torch.randn(b, 64, t * 512, device=device).type_as(audio)
        # noist=torch.randn(1,8,t*512)

        aaac=self(noist,spectrogram)[0]
        loss1,loxs2=self.lossx.stft_loss(aaac, audio)
        # aaac, opo = self.predict(spectrogram,f0, fast_sampling=False)
        loss = self.loss_fn(aaac, audio)

        accc['gad'] = aaac
        # print(loss)

        self.val_loss.append(loss)
        self.val_loss1.append(loss1)
        self.val_loss2.append(loxs2)
        accc['spectrogram']=tfff.transform(accc['audio'].detach().cpu())

        accc['spectrogramg'] = tfff.transform(aaac.detach().cpu().float())
        # self.valc.append(accc)

        self.valc.append(accc)



        return loss



if __name__ == "__main__":
    from lvc.dataset2 import from_path, from_gtzan
    from lvc.params import params

    writer = SummaryWriter("./mdsr_1000sV/", )

    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # torch.backends.cudnn.benchmark = True
    md = PL_diffwav(params)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"lagegeFDbignet_1000")
    dataset = from_path([#'./testwav/',
                        # r'K:\dataa\OpenSinger',
        r'C:\Users\autumn\Desktop\poject_all\DiffSinger\data\raw\opencpop\segments\wavs'], params)
    # dataset= from_path(['./test/', ], params, ifv=True)
    datasetv = from_path(['./test/', ], params, ifv=True)
    # md = md.load_from_checkpoint('./mdscp/sample-mnist-epoch99-99-37500.ckpt', params=params)
    # eee=torch.load('./mdscpscx/sample-mnist-epoch03-3-15024.ckpt')['state_dict']
    # del eee['diffwav.lvc_blocks.0.kernel_predictor.input_conv.0.weigh']
    # x= OrderedDict[{}]
    # e={}
    # for ii in eee:
    #     # print(ii)
    #     if ii in ['diffwav.lvc_blocks.0.kernel_predictor.input_conv.0.weight','diffwav.lvc_blocks.0.fc_t.weight','diffwav.lvc_blocks.0.fc_t.bias','diffwav.lvc_blocks.0.F0x.weight','diffwav.lvc_blocks.0.F0x.bias']:
    #         continue
    #     if ii in ['diffwav.lvc_blocks.1.kernel_predictor.input_conv.0.weight', 'diffwav.lvc_blocks.1.fc_t.weight',
    #               'diffwav.lvc_blocks.1.fc_t.bias', 'diffwav.lvc_blocks.1.F0x.weight', 'diffwav.lvc_blocks.1.F0x.bias']:
    #         continue
    #     if ii in ['diffwav.lvc_blocks.2.kernel_predictor.input_conv.0.weight', 'diffwav.lvc_blocks.2.fc_t.weight',
    #               'diffwav.lvc_blocks.2.fc_t.bias', 'diffwav.lvc_blocks.2.F0x.weight', 'diffwav.lvc_blocks.2.F0x.bias']:
    #         continue
    #     e[ii]=eee[ii]
    # md.load_state_dict(e,strict=False)
    # md = torch.compile(md)
    checkpoint_callback = ModelCheckpoint(

    # monitor = 'val/loss',

    dirpath = './mdscpscxV',

    filename = 'sample-mnist-epoch{epoch:02d}-{epoch}-{step}',

    auto_insert_metric_name = False,every_n_epochs=4,save_top_k = -1

    )
    trainer = pl.Trainer(max_epochs=1950, logger=tensorboard, devices=-1, benchmark=True, num_sanity_val_steps=4,
                        # val_check_interval=2000,
                         callbacks=[checkpoint_callback],check_val_every_n_epoch=1,
                         precision=16
                          #resume_from_checkpoint='./bignet/default/version_25/checkpoints/epoch=134-step=1074397.ckpt'
                         )
    trainer.fit(model=md, train_dataloaders=dataset, val_dataloaders=datasetv, )

