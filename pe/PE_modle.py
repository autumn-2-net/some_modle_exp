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
from typing import Tuple

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
        # 可能有bug 注意一下下
        if att_mask is not None:
            att=att.masked_fill(att_mask==0, -float('inf'))

        att = torch.softmax(att, dim=-1)
        if att_mask is not None:
            att=att.masked_fill(att_mask==0,  0.0)
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

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Positional encoding.
            PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
            PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
        Args:
            d_model (int): embedding dim.
            dropout_rate (float): dropout rate.
            max_len (int, optional): maximum input length. Defaults to 5000.
            reverse (bool, optional): Not used. Defaults to False.
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros([1, self.max_len, self.d_model])  # [B=1,T,D]
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, :, 0::2] = torch.sin(position * div_term)  # TODO
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding, (batch=1, time, ...)
        """
        assert offset + x.shape[
            1] < self.max_len, "offset: {} + x.shape[1]: {} is larger than the max_len: {}".format(
            offset, x.shape[1], self.max_len)
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.shape[1]]  # TODO
        x = x * self.xscale + pos_emb
        return x, pos_emb

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            torch.Tensor: Corresponding position encoding, #[1, T, D].
        """
        assert offset + size < self.max_len
        return self.pe[:, offset:offset + size]


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout_rate (float): Dropout rate.
            max_len (int, optional): [Maximum input length.]. Defaults to 5000.
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        assert offset + x.shape[
            1] < self.max_len, "offset: {} + x.shape[1]: {} is larger than the max_len: {}".format(
            offset, x.shape[1], self.max_len)
        x = x * self.xscale
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.shape[1]]
        return pos_emb


#上面的应该是好的

# cst =RelPositionalEncoding(128,0)
# dddd=torch.randn((2,100))
# ddd=cst(dddd)
# asts=dddd+ddd[1]
# ddd

class ph_embd(nn.Module):
    def __init__(self,config,):
        super().__init__()
        self.diaoemb = nn.Embedding(5,embedding_dim=config.n_embd,padding_idx=0)
        self.phemb = nn.Embedding(5, embedding_dim=config.n_embd,padding_idx=0) #瞎写占位的

        self.pos_emb=RelPositionalEncoding(config.n_embd,0)
        self.pith_emb=nn.Embedding(5, embedding_dim=config.n_embd,padding_idx=0)#瞎写占位的
        self.speak_emb=nn.Embedding(5, embedding_dim=config.n_embd,padding_idx=0)



        # self.diaoemb = nn.Embedding(5, embedding_dim=config.n_embd)

    def forward(self, x, diao,pith,speak):
        # getemb=self.pos_emb(x)
        return  self.diaoemb(diao)+self.phemb(x)+self.pos_emb(x)+self.pith_emb(pith)+self.speak_emb(speak)
class PE_md(pl.LightningModule):#WIP 未完成

    def __init__(self,config):
        super().__init__()
        self.bert=bert_modle(config=config,layers=config.n_lay)
        self.emb=ph_embd(config=config)

        self.opt=nn.Linear(config.n_embd, 1)
        self.sim=nn.Sigmoid()
        self.losss = nn.L1Loss()
        self.lrc=0

    def forward(self,x, diao,pith,speak,att_mask=None):
        y=self.emb(x, diao,pith,speak)
        y=self.bert(y,att_mask)
        o=self.sim(self.opt(y))
        return o

    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def training_step(self, batch, batch_idx):
        x,diao,pith,tg_f0,speak,att_mask=batch

        pf0=self.forward(x,diao,pith,speak)

        loss =self.loss(pf0, pf0)
        return loss



