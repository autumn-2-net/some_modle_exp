import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class wolo_attention(nn.Module):
    def __init__(self,dim,kernel_size,padding):
        super().__init__()
        self.linear2 = nn.Linear(dim, kernel_size**2+kernel_size)
        self.outlinear = nn.Linear(dim, dim)

        self.kernel_size=kernel_size
        self.padding=padding
    def forward(self, x):
        b,l,d=x.size()
        att=self.linear2(x)
        attention_weight,attention_bise = torch.split(att, [self.kernel_size ** 2, self.kernel_size], dim=2)
        attention_weight=torch.sin(attention_weight)
        attention_weight=attention_weight.reshape(b, l, self.kernel_size, self.kernel_size)
        x=x.transpose(1,2)
        x = F.pad(x, (self.padding, self.padding), 'constant', 0)
        x1=x.unfold(2, self.kernel_size, 1).reshape(b, l, d, self.kernel_size)
        attention_bise=attention_bise.unsqueeze(-2)
        x1=(x1@attention_weight)+attention_bise
        x1=torch.sum(x1, dim=-1)
        x1=self.outlinear(x1)
        return x1



class woloblock(nn.Module):
    def __init__(self,dim,kernel_size,dilation):
        super().__init__()
        self.act=nn.ELU()
        self.conv=nn.Conv1d(dim,dim,kernel_size=kernel_size,dilation=dilation,padding=(kernel_size + (kernel_size-1) * (dilation - 1)) // 2)
        self.att=wolo_attention(dim,kernel_size=kernel_size,padding=kernel_size//2)

    def forward(self, x):
        return self.conv(self.act(self.att(self.act(x).transpose(1,2)).transpose(1,2))) +x
    
    
# class wolob2(nn.Module):
#     def __init__(self,dim,up=2,kernel_size=3,dilation_cycle=3,dilation_lay=3):
#         super().__init__()
#         self.up = nn.ConvTranspose1d()

class woloupBlock(nn.Module):
    def __init__(self,up,indim,outdim,kernel_size,dilation):
        super().__init__()
        self.wolob=woloblock(outdim,kernel_size=kernel_size,dilation=dilation)

        s=up
        k=up*2
        p=up*1-up//2

        self.up = nn.ConvTranspose1d(indim,outdim,kernel_size=k,padding=p,stride=s)
        self.act = nn.ELU()
    def forward(self, x):
        x=self.wolob(self.up (self.act(x)))


        return x


# wwwwb=woloupBlock(4,512,256,7,1)
# xx=torch.randn(10,512,100)
# xc=wwwwb(xx)
# pass
class wolonet(nn.Module):
    def __init__(self):
        super().__init__()
        self.incov=nn.Conv1d(128,1024,kernel_size=7,padding=3)
        self.outcov = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        self.tnnh=nn.Tanh()
        self.elu = nn.ELU()
        self.w1=woloupBlock(8,1024,512,kernel_size=3,dilation=1)
        self.w2= woloupBlock(8, 512, 256, kernel_size=7, dilation=1)
        self.w3 = woloupBlock(2, 256, 128, kernel_size=11, dilation=1)
        self.w4 = woloupBlock(2, 128, 64, kernel_size=13, dilation=1)
        self.w5 = woloupBlock(2, 64, 32, kernel_size=15, dilation=1)
    def forward(self, x):
        x=self.incov(x)
        x=self.w1(x)
        x=self.w2(x)
        x=self.w3(x)
        x = self.w4(x)
        x = self.w5(x)
        x=self.tnnh(self.outcov(self.elu(x)))
        return x
#
# eeeee=wolonet()
# xxx=torch.randn(10,128,10)
# xx=eeeee(xxx)
# pass






