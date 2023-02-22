import math

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


class main_config():

    n_head = 16  #8头吧？
    n_embd =512
    n_vebs =32 #笔画长度

    n_lay=12
    embd_pdrop =0.1
    LR=0.0003
    attn_pdrop=None
    resid_pdrop=None



class MultiheadSelfAttention(nn.Module):

    # nn.MultiheadAttention()

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        # self.attn_drop = nn.Dropout(config.attn_pdrop)
        # self.resid_drop = nn.Dropout(config.resid_pdrop)
        if config.attn_pdrop is not None:
            self.attn_drop = nn.Dropout(config.attn_pdrop)
        if config.resid_pdrop is not None:
            self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.config=config
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)) #掩码 没必要bert
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None,att_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print('a')
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # if att_mask is not None: #根据原文mask的只有 图像部分 当输入补pad的时候 因为图像不会被mask 使用不需要？
        #     maskt =att_mask.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)@ k.transpose(-2, -1)
        #     one = torch.ones_like(maskt)
        #     maskt =torch.where(maskt > 0.0, one, maskt)
        #     maskt = torch.where(maskt < 0.0, one, maskt)
        #     maskt = maskt*-10000
        #     att =att+maskt

        att = torch.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        if self.config.attn_pdrop is not None:
            att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_drop(self.proj(y))
        if self.config.resid_pdrop is not None:
            y = self.resid_drop(self.proj(y))

        y=self.proj(y)
        return y

class BERT_Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadSelfAttention(config)
        # self.attn = SublinearSequential(self.attn)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            # nn.Dropout(config.embd_pdrop),

        )
        if config.embd_pdrop is not None:
            self.embd_pdrop = nn.Dropout(config.embd_pdrop)
        self.config=config


        # self.mlp = SublinearSequential(*list(self.mlp.children()))  #减小400m

    def forward(self, x,att_mask):

        x = x + self.attn(self.ln1(x),att_mask=att_mask)
        x = x + self.mlp(self.ln2(x))
        if self.config.embd_pdrop is not None:
            x=self.embd_pdrop(x)
        return x
class bert_modle(nn.Module):
    def __init__(self,config,layers=4):
        super().__init__()
        # self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([BERT_Block(config=config) for _ in range(layers)])
        # self.blocks = nn.Sequential(*[BERT_Block(config=config) for _ in range(layers)])
        # self.blocks =SublinearSequential(*list(self.blocks.children()))

    def forward(self, x,att_mask=None):
        for i in self.blocks:
            x =i(x,att_mask)
        return x
#上面的应该是好的
class ph_embd(nn.Module):
    def __init__(self,config,):
        super().__init__()
        self.diaoemb = nn.Embedding(5,embedding_dim=config.n_embd)
        self.phemb = nn.Embedding(5, embedding_dim=config.n_embd) #瞎写占位的


        self.diaoemb = nn.Embedding(5, embedding_dim=config.n_embd)

    def forward(self, x, diao,):
        return  self.diaoemb(diao)+self.phemb(x)
class PE_md(pl.LightningModule):#WIP 未完成

    def __init__(self,config):
        super().__init__()
        self.bert=bert_modle(config=config,layers=config.n_lay)
        self.emb=ph_embd(config=config)

        self.opt=nn.Linear(config.n_embd, 1)
        self.sim=nn.Sigmoid()

    def forward(self,x):
        y=self.emb(x)
        y=self.bert(y)
        o=self.sim(self.opt(y))
        return o



