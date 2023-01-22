import random
from glob import glob

import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import torch

import pytorch_lightning as pl
import torch.nn.functional as F

from math import sqrt
import numpy as np
from pytorch_lightning import loggers as pl_loggers
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data


#
# astc=torch.Tensor(1,40)
# a=0
#
#
# cov1d=nn.Conv1d(1,1,kernel_size=6,stride=2)
# cov1d2=nn.Conv1d(1,1,kernel_size=6,stride=2)
# cov1d3=nn.Conv1d(1,1,kernel_size=6,stride=2)
# upc=nn.ConvTranspose1d(1,1,kernel_size=6,stride=2)
# upc2=nn.ConvTranspose1d(1,1,kernel_size=6,stride=2)
# upc3=nn.ConvTranspose1d(1,1,kernel_size=6,stride=2)
#
# sss=cov1d(astc)
# sss2=cov1d2(sss)
# sss3=cov1d3(sss2)
# ccc=upc(sss3)
# ccc2=upc2(ccc)
# accc3=upc3(ccc2)
#
# a=0
# ccd=[]
#
# for i in tqdm(range(44100*3)):
#     i=i+36
#     astc = torch.Tensor(1, i).cuda()
#     sz1=astc.size()
#     sss = cov1d(astc)
#     sss2 = cov1d2(sss)
#     sss3 = cov1d3(sss2)
#     ccc = upc(sss3)
#     ccc2 = upc2(ccc)
#     accc3 = upc3(ccc2)
#     sz2=accc3.size()
#     if sz1==sz2:
#
#         ccd.append(i)
#
# print(ccd)

def up_pad(x):
    if x < 36:
        return 36, 36 - x
    d = x - 36
    if d == 0:
        return 36, 0

    if d % 8 == 0:
        return x, 0
    return x + 8 - d % 8, 8 - d % 8


class en_cov(nn.Module):
    def __init__(self):
        super().__init__()

        self.cov1d = nn.Conv1d(1, 2, kernel_size=6, stride=2)
        self.cov1d2 = nn.Conv1d(2, 4, kernel_size=6, stride=2)
        self.cov1d3 = nn.Conv1d(4, 6, kernel_size=6, stride=2)

    def forward(self, x):
        x = self.cov1d(x)
        x = silu(x)
        x = self.cov1d2(x)
        x = silu(x)
        x = self.cov1d3(x)
        x = silu(x)
        return x


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class de_cov(nn.Module):
    def __init__(self):
        super().__init__()

        self.upc = nn.ConvTranspose1d(6, 4, kernel_size=6, stride=2)
        self.upc2 = nn.ConvTranspose1d(4, 2, kernel_size=6, stride=2)
        self.upc3 = nn.ConvTranspose1d(2, 1, kernel_size=6, stride=2)

    def forward(self, x):
        x = self.upc(x)
        x = silu(x)
        x = self.upc2(x)
        x = silu(x)
        x = self.upc3(x)
        x = silu(x)
        return x


class vec_pl(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ec = en_cov()
        self.dc = de_cov()

        self.grad_norm = 0
        self.loss = nn.L1Loss()
        self.step = 0

    def forward(self, x):
        x = self.ec(x)
        x = self.dc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.002)
        return optimizer

    def on_after_backward(self):
        self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(), None or 1e9)

    def training_step(self, batch, batch_idx):
        x, p = batch
        y = self.forward(x)

        # b1 = beatch_cut(y, p)
        # b2 = beatch_cut(x, p)
        b1 = y
        b2 = x
        loss = self.loss(b1, b2)

        self.step = self.step + 1
        if self.step % 500 == 0:
            self._write_summary(self.step, beatch_cut(x, p), beatch_cut(y, p), loss)

        return loss

    def _write_summary(self, step, insound, outs, loss):  # log 器
        tensorboard = self.logger.experiment
        # writer = tensorboard.SummaryWriter
        writer = SummaryWriter("./mdss/", purge_step=step)
        # writer = tensorboard
        writer.add_audio('feature/audio', insound[0], step, sample_rate=44100)
        writer.add_audio('out/audio', outs[0], step, sample_rate=44100)

        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer


def beatch_cut(bt, pdd: list):
    act = []

    blcn = bt.size()[0]
    blc = torch.chunk(bt, blcn, dim=0)
    for idx, i in enumerate(blc):
        icp = -int(pdd[idx])
        act.append(i.squeeze(0).squeeze(0)[:icp].unsqueeze(0).unsqueeze(0))

    return tuple(act)


class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            print(paths, path)
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]

        signal, _ = torchaudio.load(audio_filename)

        return {
            'audio': signal[0],

        }


def make_time(time):
    upc = up_pad(time)
    if random.randint(1, 10) == 4:
        return time
    if upc[1] == 0:
        return time - random.randint(1, 7)
    full = upc[0]
    return full - random.randint(1, 7)


class Collator:

    def collate(self, minibatch):

        time = int(random.randint(1, 2) * random.random() * 44100)
        pplsdc = []

        for record in minibatch:
            rpct = make_time(time)
            pdc = up_pad(rpct)
            # Filter out records that aren't long enough.
            if len(record['audio']) < 44100 * 2.5:
                del record['audio']
                continue

            start = random.randint(0, record['audio'].shape[-1] - rpct)
            end = start + rpct
            record['audio'] = record['audio'][start:end]
            record['audio'] = np.pad(record['audio'], (0, pdc[1]), mode='constant')
            pplsdc.append(pdc[1])

        # audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        cdts = []
        for iee in minibatch:
            if 'audio' in iee:
                cdts.append(torch.from_numpy(iee['audio']).unsqueeze(0))

        sss = torch.cat(tuple(cdts), dim=0).unsqueeze(1)

        return sss, torch.tensor(pplsdc)

        #
        # return {
        #     'audio': torch.from_numpy(audio),
        #     'spectrogram': torch.from_numpy(spectrogram),
        # }

    # for gtzan


def from_path(data_dirs, is_distributed=False):
    dataset = ConditionalDataset(data_dirs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=48,
        collate_fn=Collator().collate,
        shuffle=not is_distributed,
        # num_workers=os.cpu_count(),
        num_workers=2,

        pin_memory=True,
        drop_last=False  # flase hao hai shi?
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    md = vec_pl()
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="", name='sdxc')
    dataset = from_path([r'C:\Users\autumn\Desktop\poject_all\vcoder\testwav'], )

    trainer = pl.Trainer(max_epochs=100, logger=tensorboard, gpus=1)
    trainer.fit(model=md, train_dataloader=dataset)
