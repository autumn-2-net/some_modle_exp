# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, paths,ikvv,blcl='black.txt'):
        super().__init__()
        self.filenames = []
        with open(blcl,'r',encoding='utf-8') as f:
            ddddf=f.read().strip().split('\n')
        for path in paths:
            print(paths, path)
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)
        if not ikvv:
            self.filenames=self.filenames
        for i in ddddf:
            if i in self.filenames:
                self.filenames.remove(i)
        pass
        # else:
        #     eee=self.filenames.copy()
        #     o=[]
        #     for i in eee:



    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        spec_filename = f'{audio_filename}.spec.npy'
        signal, _ = torchaudio.load(audio_filename)
        spectrogram = np.load(spec_filename)
        f0_n = f'{audio_filename}.f0.npy'
        uv_n = f'{audio_filename}.uv.npy'

        uv=np.load(uv_n)
        f0=np.load(f0_n)
        # spectrogramf0 = np.load(f'{audio_filename}.f0.npy')
        # return signal[0],spectrogram.T
        return {
            'audio': signal[0],
            'spectrogram': spectrogram.T,#'f0':spectrogramf0
            'f0': f0,
            'uv': uv
        }


class UnconditionalDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        spec_filename = f'{audio_filename}.spec.npy'
        signal, _ = torchaudio.load(audio_filename)
        return {
            'audio': signal[0],
            'spectrogram': None
        }


class Collator:
    def __init__(self, params,ifv):
        self.params = params
        self.ifv=ifv

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        if  self.ifv:
            crop_mel_frames=self.params.val_crop_mel_frames
        else:
            crop_mel_frames = self.params.crop_mel_frames

        for record in minibatch:
            if self.params.unconditional:
                # Filter out records that aren't long enough.
                if len(record['audio']) < self.params.audio_len:
                    del record['spectrogram']
                    del record['audio']
                    # del record['f0']
                    # del record['spectrogram']
                    # del record['audio']
                    del record['f0']
                    del record['uv']
                    continue

                start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
                end = start + self.params.audio_len
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
            else:
                # Filter out records that aren't long enough.
                if len(record['spectrogram']) < crop_mel_frames:
                    del record['spectrogram']
                    del record['audio']
                    del record['f0']
                    del record['uv']
                    # del record['f0']
                    continue

                start = random.randint(0, record['spectrogram'].shape[0]-1 - crop_mel_frames)
                end = start + crop_mel_frames
                if self.ifv:
                    record['spectrogram'] = record['spectrogram'].T
                    record['uv'] = record['uv']
                    record['f0'] = record['f0']
                    # record['f0'] = record['f0']
                else:
                    record['spectrogram'] = record['spectrogram'][start:end].T
                    record['uv'] = record['uv'][start:end]
                    record['f0'] = record['f0'][start:end]
                    # record['f0'] = record['f0'][start:end]
                start *= samples_per_frame
                end *= samples_per_frame
                if self.ifv:
                    record['audio'] = record['audio']
                    record['audio'] = np.pad(record['audio'], (0, (len(record['spectrogram'].T)*samples_per_frame ) - len(record['audio'])),
                                             mode='constant')
                    pass
                else:
                    # record['spectrogram'] = record['spectrogram'][start:end].T
                    record['audio'] = record['audio'][start:end]
                    record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        if self.params.unconditional:
            return {
                'audio': torch.from_numpy(audio),
                'spectrogram': None,
            }
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        f0 = np.stack([record['f0'] for record in minibatch if 'f0' in record])
        uv = np.stack([record['uv'] for record in minibatch if 'uv' in record])
        # f0f=np.stack([record['f0'] for record in minibatch if 'f0' in record])
        return torch.from_numpy(audio),torch.from_numpy(spectrogram),torch.from_numpy(f0).type_as(torch.from_numpy(audio)),torch.from_numpy(uv).type_as(torch.from_numpy(audio))#torch.from_numpy(f0f)
        # return {
        #     'audio': torch.from_numpy(audio),
        #     'spectrogram': torch.from_numpy(spectrogram),
        # }

    # for gtzan
    def collate_gtzan(self, minibatch):
        ldata = []
        mean_audio_len = self.params.audio_len  # change to fit in gpu memory
        # audio total generated time = audio_len * sample_rate
        # GTZAN statistics
        # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
        # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
        # sample rate of all is 22050
        for data in minibatch:
            if data[0].shape[-1] < mean_audio_len:  # pad
                data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
            elif data[0].shape[-1] > mean_audio_len:  # crop
                start = random.randint(0, data[0].shape[-1] - mean_audio_len)
                end = start + mean_audio_len
                data_audio = data[0][:, start:end]
            else:
                data_audio = data[0]
            ldata.append(data_audio)
        audio = torch.cat(ldata, dim=0)
        return {
            'audio': audio,
            'spectrogram': None,
        }


def from_path(data_dirs, params, is_distributed=False,ifv=False):
    if params.unconditional:
        dataset = UnconditionalDataset(data_dirs)
    else:  # with condition
        dataset = ConditionalDataset(data_dirs,ifv)
    bs=params.batch_size
    if ifv:
        bs=1
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        collate_fn=Collator(params,ifv).collate,
        shuffle=not ifv,
        # num_workers=os.cpu_count(),
        num_workers=params.num_cpu,prefetch_factor=4,

        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=params.pin_memory,
        drop_last=params.drop_last  # flase hao hai shi?
    )


def from_gtzan(params, is_distributed=False):
    dataset = torchaudio.datasets.GTZAN('./data', download=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate_gtzan,
        shuffle=not is_distributed,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True)
