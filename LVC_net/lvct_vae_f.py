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

from model.discriminator import Discriminator
from scheduler import V3LSGDRLR
from SWN import SwitchNorm2d
from T_lvc import LVCNetGenerator, ParallelWaveGANDiscriminator
from losses import PWGLoss
from lvc import tfff

# tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
# from fd.modules import DiffusionDBlock, TimeAware_LVCBlock
# from fd.sc import V3LSGDRLR



class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class res2dlay(nn.Module):
    def __init__(self,cd,dilation):
        super().__init__()
        assert cd%4==0
        c=cd//4
        self.cov2dk3=nn.Conv2d(c,c*2,kernel_size=3,dilation=dilation,padding=(3 + 2 * (dilation - 1)) // 2)
        self.cov2dk5 = nn.Conv2d(c,c*2,kernel_size=5,dilation=dilation,padding=(5 + 4 * (dilation - 1)) // 2)
        self.cov2dk7 = nn.Conv2d(c,c*2,kernel_size=7,dilation=dilation,padding=(7 + 6 * (dilation - 1)) // 2)
        self.cov2dk9 = nn.Conv2d(c,c*2,kernel_size=9,dilation=dilation,padding=(9 + 8 * (dilation - 1)) // 2)
        self.Glu=GLU(1)

    def forward(self, x:torch.Tensor):
        inputs=x.chunk(4,dim=1)
        x1=self.Glu(self.cov2dk3(inputs[0]))
        x2 = self.Glu(self.cov2dk5(inputs[1]))
        x3 = self.Glu(self.cov2dk7(inputs[2]))
        x4= self.Glu(self.cov2dk9(inputs[3]))
        out=torch.cat([x1,x2,x3,x4],dim=1)
        return out

class dres2dnlocke(nn.Module):
    def __init__(self,dim,Dcycle=2,lay=4):
        super().__init__()
        self.incov=nn.Conv2d(1,dim*2,kernel_size=1)
        self.Glu=GLU(1)
        self.model=nn.ModuleList()
        for i in range(lay):
            self.model.append(res2dlay(dim, 2 ** (i % Dcycle)))
        self.outc=nn.Conv2d(dim,2,kernel_size=1)

    def forward(self,x:torch.Tensor): #b ct->bct
        x=self.Glu(self.incov(x.unsqueeze(1)))#b ct->b dim c t
        for i in self.model:
            x=x+i(x)
        return self.Glu(self.outc(x)).squeeze(1)#b dim c t->bct


class res1dlay(nn.Module):
    def __init__(self,cd,dilation):
        super().__init__()
        assert cd%4==0
        c=cd//4
        self.cov2dk3=nn.Conv1d(c,c*2,kernel_size=3,dilation=dilation,padding=(3 + 2 * (dilation - 1)) // 2)
        self.cov2dk5 = nn.Conv1d(c,c*2,kernel_size=5,dilation=dilation,padding=(5 + 4 * (dilation - 1)) // 2)
        self.cov2dk7 = nn.Conv1d(c,c*2,kernel_size=7,dilation=dilation,padding=(7 + 6 * (dilation - 1)) // 2)
        self.cov2dk9 = nn.Conv1d(c,c*2,kernel_size=9,dilation=dilation,padding=(9 + 8 * (dilation - 1)) // 2)
        self.Glu=GLU(1)

    def forward(self, x:torch.Tensor):
        inputs=x.chunk(4,dim=1)
        x1=self.Glu(self.cov2dk3(inputs[0]))
        x2 = self.Glu(self.cov2dk5(inputs[1]))
        x3 = self.Glu(self.cov2dk7(inputs[2]))
        x4= self.Glu(self.cov2dk9(inputs[3]))
        out=torch.cat([x1,x2,x3,x4],dim=1)
        return out

class dres1dnlocke(nn.Module):
    def __init__(self,dim,Dcycle=2,lay=4):
        super().__init__()
        # self.incov=nn.Conv1d(1,dim*2,kernel_size=1)
        self.Glu=GLU(1)
        self.model=nn.ModuleList()
        for i in range(lay):
            self.model.append(res1dlay(dim, 2 ** (i % Dcycle)))
        # self.outc=nn.Conv1d(dim,2,kernel_size=1)

    def forward(self,x:torch.Tensor): #b ct->bct

        for i in self.model:
            x=x+i(x)
        return x


class downlay(nn.Module):
    def __init__(self,indim,outdim,downx=2):
        super().__init__()

        self.downconv=nn.Conv1d(indim,outdim*2,kernel_size=downx*2,stride=downx,padding=(downx//2))
        self.glu=GLU(1)
    def forward(self, x):
        return self.glu(self.downconv(x))







class encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.incov=nn.Conv1d(1,64,kernel_size=1)
        self.glu=GLU(1)
        # self.res1=dres1dnlocke(8)
        self.down1=downlay(32,64,downx=4) #4
        self.res1d1=dres1dnlocke(32,Dcycle=8,lay=8)
        self.res2d1=dres2dnlocke(4,Dcycle=2,lay=4)

        self.down2=downlay(64,96,downx=8) #16
        self.res1d2=dres1dnlocke(48,Dcycle=4,lay=4)
        self.res2d2=dres2dnlocke(4,Dcycle=2,lay=4)

        self.down3=downlay(96,128,downx=4)#64
        self.res1d3=dres1dnlocke(64,Dcycle=4,lay=4)
        self.res2d3=dres2dnlocke(12,Dcycle=2,lay=4)

        self.down4=downlay(128,256,downx=4)#256
        self.res1d4=dres1dnlocke(128,Dcycle=2,lay=4)
        self.res2d4=dres2dnlocke(32,Dcycle=2,lay=4)

        # self.down5 = downlay(256, 256, downx=2)#512
        # self.res1d5 = dres1dnlocke(128, Dcycle=3, lay=6)
        # self.res2d5 = dres2dnlocke(16, Dcycle=2, lay=4)

        self.outcov = nn.Conv1d(256, 256, kernel_size=1)

    def forward(self, x:torch.Tensor):
        x1,x2=self.down1(self.glu(self.incov(x))).chunk(2, dim=1)
        x1=self.res1d1(x1)
        x2 = self.res2d1(x2)

        x1, x2 = self.down2(torch.cat([x1,x2],dim=1)).chunk(2, dim=1)
        x1=self.res1d2(x1)
        x2 = self.res2d2(x2)

        x1, x2 = self.down3(torch.cat([x1,x2],dim=1)).chunk(2, dim=1)
        x1=self.res1d3(x1)
        x2 = self.res2d3(x2)

        x1, x2 = self.down4(torch.cat([x1,x2],dim=1)).chunk(2, dim=1)
        x1=self.res1d4(x1)
        x2 = self.res2d4(x2)

        # x1, x2 = self.down5(torch.cat([x1,x2],dim=1)).chunk(2, dim=1)
        # x1=self.res1d5(x1)
        # x2 = self.res2d5(x2)

        x= self.glu(self.outcov(torch.cat([x1,x2],dim=1)))
        return x
#
# rends=encode().cuda()
# ppp=torch.randn(10,1,512*62).cuda()
# outt=rends(ppp)
# print(outt.size())



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


class covD(nn.Module):
    def __init__(self,C,D):
        super().__init__()
        self.Glu=GLU(1)
        self.n1=SwitchNorm2d(C)
        self.n2 = SwitchNorm2d(C)
        self.n3 = SwitchNorm2d(C)
        self.n4 = SwitchNorm2d(C)

        self.C1=nn.Conv2d(C,C*2,kernel_size=3,padding=(3 + 2 * (D - 1)) // 2,dilation=D)
        self.C2 = nn.Conv2d(C, C * 2, kernel_size=5, padding=(5 + 4 * (D - 1)) // 2, dilation=D)
        self.C3 = nn.Conv2d(C, C * 2, kernel_size=7, padding=(7 + 6 * (D - 1)) // 2, dilation=D)
        self.C4 = nn.Conv2d(C, C * 2, kernel_size=9, padding=(9 + 8 * (D - 1)) // 2, dilation=D)
    def forward(self, x):
        x=x+self.n1(self.Glu(self.C1(x)))
        x = x + self.n2(self.Glu(self.C2(x)))
        # x = x + self.n3 (self.Glu(self.C3(x)))
        # x = x + self.n4(self.Glu(self.C4(x)))
        return x

class UpconD(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 8, kernel_size=1)
        self.UP = torch.nn.ConvTranspose2d(4, 8, [3, 32], stride=[1, 2], padding=[1, 15])
        self.Glu = GLU(1)
        self.c3 = torch.nn.Conv2d(4, 2, kernel_size=1)

        # self.Modl = []
        # self.net=nn.Sequential()
        self.Modl = []

        for i in range(3):
            self.Modl.append(covD(4, 2 ** (i % 3)))
        self.net = nn.Sequential(*self.Modl)

        self.Modl = []

        for i in range(3):
            self.Modl.append(covD(4, 2 ** (i % 3)))
        self.net1 = nn.Sequential(*self.Modl)


    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x=self.Glu (self.c1(x))
        # x=self.net(x)

        x = self.Glu (self.UP(x))
        x = self.net1(x)
        x =self.Glu(self.c3(x))
        # x = F.leaky_relu(x, 0.4)
        # x=F.leaky_relu(self.c1(x), 0.4)+x
        # x = F.leaky_relu(self.c2(x), 0.4) + x
        # x = F.leaky_relu(self.cc2(x), 0.4) + x
        # x = F.leaky_relu(self.ccc2(x), 0.4) + x
        # x = F.leaky_relu(self.c3(x), 0.4)

        # x = self.conv2(x)
        # x = F.leaky_relu(x, 0.4)
        spectrogram = torch.squeeze(x, 1)
        return spectrogram



class PL_diffwav(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.diffwav = LVCNetGenerator(in_channels=16,
                 out_channels=1,
                 inner_channels=18,#不知道10压不压住 441
                 cond_channels=128,
                 cond_hop_length=256,
                 lvc_block_nums=3,#这个有点问题
                 lvc_layers_each_block=10, #这个不能动
                 lvc_kernel_size=7,# 9或者7较好
                 kpnet_hidden_channels=96,
                 kpnet_conv_size=3,
                 dropout=0.0,)
        self.D=ParallelWaveGANDiscriminator()
        self.D2=Discriminator(mrd_resolutions=[(1024, 120, 600), (2048, 240, 1200), (512, 50, 240), (4096, 128, 4096)])
        self.encode=encode()

        # self.model_dir = model_dir
        # self.model = model
        # self.dataset = dataset
        # self.optimizer = optimizer
        # self.params = params
        # self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        # self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True
        self.lossx=PWGLoss(stft_loss_params={
                  'fft_sizes':[2048,1024, 2048, 512,128,2048,2048,4096], #fft 4096 有问题
                  'hop_sizes':[256 ,120 , 240 , 50 ,25 ,256, 512 ,128],
                'win_lengths':[1024,600 , 1200, 240,64 ,512 ,1024,4096],
                 'window':"hann_window"})



        self.UP=UpconD()

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
        self.automatic_optimization = False
        self.sidx=0

    def forward(self, audio,noise):
        spectrogram=self.encode(audio)
        spectrogram = self.UP(spectrogram)

        # x = self.conv2(x)
        # x = F.leaky_relu(x, 0.4)



        return self.diffwav(noise,spectrogram)

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
        opt_g, opt_d = self.optimizers()




        audio = accc['audio'].unsqueeze(1)
        spectrogram = accc['spectrogram']
        device = audio.device

        b, c, t = spectrogram.size()
        noist = torch.randn(b, 16, t * 512,device=device).type_as(audio)

        aaac = self(audio,noist)
        ####################
        #T D
        ####################

        TTT=self.D(audio)
        FFF = self.D(aaac.detach())

        mrdTL=0.0
        mrdFL = 0.0
        mpdTL = 0.0
        mpdFL = 0.0
        mrdT,mpdT=self.D2(audio)
        mrdF, mpdF = self.D2(aaac.detach())

        mrdll=len(mrdT)
        mpdll=len(mpdF)

        for i1, i2 in zip(mrdT,mrdF):
            mrdTL +=torch.nn.MSELoss()(i1,  torch.ones_like( i1 ))
            mrdFL+=torch.nn.MSELoss()(i2,  torch.zeros_like(i2))
        mrdTL=mrdTL/mrdll
        mrdFL=mrdFL/mrdll

        for i1, i2 in zip(mpdT,mpdF):
            mpdTL +=torch.nn.MSELoss()(i1,  torch.ones_like( i1 ))
            mpdFL+=torch.nn.MSELoss()(i2,  torch.zeros_like(i2))
        mpdTL=mpdTL/mpdll
        mpdFL=mpdFL/mpdll

        lopp=mrdTL+mrdFL+mpdTL+mpdFL



            # loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
            # loss_d += torch.mean(torch.pow(score_fake, 2))

        ################################
        # D优化
        ####################################
        # TTTL,FFFL=self.lossx.discriminator_loss(TTT,FFF)
        TTTL=torch.nn.MSELoss()(TTT,  torch.ones_like( TTT ))
        FFFL = torch.nn.MSELoss()(TTT,  torch.zeros_like(FFF))

        losssx=TTTL+FFFL+lopp

        opt_d.zero_grad()
        self.manual_backward(losssx)

        # print(FFF.grad,TTT,FFF,)
        opt_d.step()


        # G Train

        NNN=self.D(aaac)
        GmrdL=0.0
        GmpdL=0.0

        Gmrd, Gmpd = self.D2(aaac)
        Gmrdn=len(Gmrd)
        Gmpdn = len(Gmpd)
        for i1 in Gmrd:
            GmrdL +=torch.nn.MSELoss()(i1,  torch.ones_like( i1 ))
        for i1 in Gmpd:
            GmpdL +=torch.nn.MSELoss()(i1,  torch.ones_like( i1 ))

        GmrdL=GmrdL/Gmrdn
        GmpdL=GmpdL/Gmpdn
        mixgl=GmrdL+GmpdL


        # GDL=self.lossx.adversarial_loss(NNN)
        GDL= torch.nn.MSELoss()(NNN,  torch.ones_like(NNN))
        # print(FFFL, torch.zeros_like(FFF),GDL,torch.ones_like(NNN) )



        loss=self.loss_fn(aaac,audio)
        loss1,loxs2=self.lossx.stft_loss(aaac, audio)
        # lopp=mrdTL+mrdFL+mpdTL+mpdFL
        if (1+ self.global_step) %50==0 or self.global_step %50==0 or self.global_step %51==0 or self.global_step ==0 or self.global_step ==1:
            self._write_summary(self.global_step, accc, loss,aaac,loss1,loxs2,FFFL,TTTL,GDL,mrdTL,mrdFL,mpdTL,mpdFL,GmrdL,GmpdL)
        # self._write_summary(self.global_step, accc, loss, aaac, loss1, loxs2, FFFL, TTTL, GDL)

        loss=((loss1+loxs2+loss)*100+GDL*1+mixgl)/100

        opt_g.zero_grad()

        self.manual_backward(loss)
        opt_g.step()






        # return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        opt_g = torch.optim.AdamW((list(self.diffwav.parameters())+list(self.encode.parameters())+list(self.UP.parameters())), lr=self.params.learning_rate)
        opt_d = torch.optim.AdamW((list(self.D.parameters())+list(self.D2.parameters())), lr=self.params.learning_rate)
        # lt = {
        #     "scheduler": V3LSGDRLR(optimizer,),  # 调度器
        #     "interval": 'step',  # 调度的单位，epoch或step
        #     #"frequency": self.params.frequency,  # 调度的频率，多少轮一次
        #     "reduce_on_plateau": False,  # ReduceLROnPlateau
        #     "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
        #     "strict": False  # 如果没有monitor，是否中断训练
        # }
        return [opt_g, opt_d]
            # {"optimizer": optimizer,
            #     # "lr_scheduler": lt
            #     }

    # def on_after_backward(self):
    #     self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm or 1e9)

    # train
    # mixgl = GmrdL + GmpdL
    def _write_summary(self, step, features, loss,pre_audio,sc_loss, mag_loss,DFloss,DTloss,G_ganloss,mrdTL,mrdFL,mpdTL,mpdFL,GmrdL,GmpdL):  # log 器
        # tensorboard = self.logger.experiment
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)

        # if not self.params.unconditional:
        self.sidx+=1

        if self.sidx>=15:
            self.sidx=0
            writer.add_audio('T_' + '' + '/audio_P', pre_audio[0], step,
                             sample_rate=self.params.sample_rate)
            acccX = tfff.transform(pre_audio[0].detach().cpu().float())[0]
            mel = self.plot_mel([

                acccX,
            ],
                ["PSpectrogram"],
            )
            # writer.add_figure('val_' + str(self.global_step) + '/spectrogram', mel, idx)
            writer.add_figure('feature/spectrogram',mel , step)
        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/sc_loss', sc_loss, step)
        writer.add_scalar('train/mag_loss', mag_loss, step)

        writer.add_scalar('train/DFloss', DFloss, step)
        writer.add_scalar('train/DTloss', DTloss, step)
        writer.add_scalar('train/G_ganloss', G_ganloss, step)

        writer.add_scalar('trainD/mrdTL', mrdTL, step)
        writer.add_scalar('trainD/mrdFL', mrdFL, step)
        writer.add_scalar('trainD/mpdTL', mpdTL, step)

        writer.add_scalar('trainD/mpdFL', mpdFL, step)
        writer.add_scalar('trainD/GmrdL', GmrdL, step)
        writer.add_scalar('trainD/GmpdL', GmpdL, step)



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
        with torch.no_grad():
            noist = torch.randn(b, 16, t * 512, device=device).type_as(audio)
            # noist=torch.randn(1,8,t*512)

            aaac=self(audio.unsqueeze(1),noist)[0]
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

    writer = SummaryWriter("./mdsr_1000sVGvae/", )

    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # torch.backends.cudnn.benchmark = True
    md = PL_diffwav(params)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"lagegeFDbignet_1000")
    dataset = from_path([#'./testwav/',
                        r'K:\dataa\OpenSinger',
        r'C:\Users\autumn\Desktop\poject_all\DiffSinger\data\raw\opencpop\segments\wavs'
        ,r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\m4singer'
        ,r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\DAta2',r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\Data3'], params)

    # dataset = from_path([  # './testwav/',
    #
    #     r'./ttttttttttt'], params)

    # dataset= from_path(['./test/', ], params, ifv=True)
    datasetv = from_path(['./test/',r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\test2','./test3' ], params, ifv=True)
    # md = md.load_from_checkpoint('./mdscp/sample-mnist-epoch99-99-37500.ckpt', params=params)

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
    # eee = torch.load(r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\mdscpscxV\sample-mnist-epoch03-3-15024.ckpt')[
    #     'state_dict']
    # md.load_state_dict(eee,strict=False)
    # md = torch.compile(md)
    checkpoint_callback = ModelCheckpoint(

    # monitor = 'val/loss',

    dirpath = './mdscpscxVGXvae',

    filename = 'sample-mnist-epoch{epoch:02d}-{epoch}-{step}',

    auto_insert_metric_name = False,every_n_epochs=2,save_top_k = -1

    )
    aaaa=list(md.parameters())
    trainer = pl.Trainer(max_epochs=1950, logger=tensorboard, devices=-1, benchmark=True, num_sanity_val_steps=-1,
                        # val_check_interval=2000,
                         callbacks=[checkpoint_callback],  check_val_every_n_epoch=1,
                         #val_check_interval=500,
                         precision=16
                          #resume_from_checkpoint='./bignet/default/version_25/checkpoints/epoch=134-step=1074397.ckpt'
                         )
    trainer.fit(model=md, train_dataloaders=dataset, val_dataloaders=datasetv,
                # ckpt_path=r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\mdscpscxVGX\sample-mnist-epoch23-23-242244.ckpt'
                )

