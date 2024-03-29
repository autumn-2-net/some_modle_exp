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
import pickle

import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser

from tqdm import tqdm

from diffwave.params import AttrDict, params as base_params
from diffwave.params import  params as sss
#from diffwave.model import DiffWave
from PL_diffwav_modle_z import PL_diffwav

models = {}

def predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
  # Lazy load model.
  if not model_dir in models:
   # if os.path.exists(f'{model_dir}/ccci3.cpt'):
     # checkpoint = torch.load(f'{model_dir}/ccci3.cpt')
     # print(checkpoint)
   # else:
     # checkpoint = torch.load(model_dir)
    # model = DiffWave(AttrDict(base_params)).to(device)
    # from diffwave.params import params
   # model = PL_diffwav(params=base_params).load_from_checkpoint(r"./default/version_57/checkpoints/epoch=23-step=253292.ckpt",params=base_params)

    model = PL_diffwav(params=base_params).load_from_checkpoint(
      r"./bignet_1000/lightning_logs/version_5/checkpoints/epoch=13-step=64879.ckpt", params=base_params)

    model=model
    model=model.to(device)
    # model.load_state_dict(checkpoint)
    # checkpoint = torch.load(r"C:\Users\autumn\Desktop\poject_all\vcoder\default\version_50\checkpoints\epoch=13-step=148228.ckpt")
    # model.load_state_dict(checkpoint)
    model.eval()
  #   models[model_dir] = model
  #
  # model = models[model_dir]
  # model.params.override(params)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(params.noise_schedule)
    inference_noise_schedule = np.array(params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], params.hop_samples * spectrogram.shape[-1], device=device)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    for n in tqdm(range(len(alpha) - 1, -1, -1)):
      # print(n)
    # for n in range(len(alpha) - 1, -1, -1):  #扩散过程
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, params.sample_rate


def main(args):
  #if args.spectrogram_path:
  #  spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
#  else:
 #   spectrogram = None
  llll=[]
  n=0
  for i in torch.load(args.spectrogram_path):
    spectrogram = i['mel']
    spectrogram=torch.transpose(spectrogram,1,2)/0.434294
    audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
    torchaudio.save(args.output+str(n)+'.wav', audio.cpu(), sample_rate=sr)
    n+=1

    llll.append(audio)
  audio = torch.cat(tuple(llll),1)

  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  #ggg=np.load('./t/0001.wav.mel.npy')
 # for i in ggg:
 #   ass=ggg[i]
  dddd=torch.load('./o2/左手指月.mel.pt')
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--model_dir',default='./md',
      help='directory containing a trained model (or full path to weights.pt file)')
  #parser.add_argument('--spectrogram_path', '-s',default='./t/凯尔特史诗配乐Vindsvept Ep.2 _2022.7.14更新12P_-p31-A Voice in the Wind-16_converted.wav.spec.npy',
   #   help='path to a spectrogram file generated by diffwave.preprocess')
  # parser.add_argument('--spectrogram_path', '-s', default='./t/0001.wav.mel.npy',
  #                     help='path to a spectrogram file generated by diffwave.preprocess')
  # parser.add_argument('--spectrogram_path', '-s', default='./test/2099003695.wav.spec.npy',
  #                     help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--spectrogram_path', '-s', default='./o2/左手指月.mel.pt',
                      help='path to a spectrogram file generated by diffwave.preprocess')
  # parser.add_argument('--spectrogram_path', '-s', default='./t/50.wav.spec.npy',
  #                     help='path to a spectrogram file generated by diffwave.preprocess')

  parser.add_argument('--output', '-o', default='./o2/nnnzybybzss.wav',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure',
                      #default = True
                      )
  main(parser.parse_args())
