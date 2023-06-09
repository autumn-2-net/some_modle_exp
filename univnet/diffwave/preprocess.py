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
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob

from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from diffwave.params import params
def get_mel_from_audio(
    # audio: torch.Tensor,
    sample_rate=44100,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    f_min=40,
    f_max=16000,
    n_mels=128,
    center=True,
    power=1.0,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="slaney",
) :


    # assert audio.ndim == 2, "Audio tensor must be 2D (1, n_samples)"
    # assert audio.shape[0] == 1, "Audio tensor must be mono"

    transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        center=center,
        power=power,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
    )#.to(audio.device)
    return transform

def transform(filename):
  audio, sr = T.load(filename)
  audio = torch.clamp(audio[0], -1.0, 1.0)

  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  mel_args = {
      'sample_rate': sr,
      'win_length': params.win_length,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': params.f_min,
      'f_max': params.f_max,
      'n_mels': params.n_mels,
      'power': params.pre_power,
      'normalized': False,
      'norm' : "slaney",
  'mel_scale' : "slaney",
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)
  mel_spec_transform=get_mel_from_audio()

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    # spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram=torch.log(torch.clamp(spectrogram, min=1e-5))
    # spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))

def myuse(pathchs):
    filenames = glob(f'{pathchs}/**/*.wav', recursive=True)
    with ProcessPoolExecutor(max_workers=6) as executor:
        list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))

if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
