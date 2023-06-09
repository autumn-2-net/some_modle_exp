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
from WavLM import WavLM, WavLMConfig
# tensorboard = pl_loggers.TensorBoardLogger(save_dir="")

class wlm(pl.LightningModule):
    def __init__(self,cfg,w):
        super().__init__()
        self.lmmodel = WavLM(cfg)
        self.lll=nn.Linear(cfg.encoder_embed_dim,w)

    def forward(self,x,mask=None,maskt=True):
        f,_=self.lmmodel(x,padding_mask=mask,mask=maskt)
        o=self.lll(f)
        return o

    def training_step(self, batch,batch_idx) :
        pass



checkpoint = torch.load('WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
checkpoint = torch.load('WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

eee=wlm(cfg)
eee

