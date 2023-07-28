



import torch

import torchaudio.transforms as TT



from lvc.params import params
def transform(audio):

  audio = torch.clamp(audio, -1.0, 1.0)


  mel_args = {
      'sample_rate': params.sample_rate,
      'win_length': 4096,
      'hop_length': 128,
      'n_fft': 4096,
      'f_min': 20.0,
      'f_max': params.sample_rate / 2.0,
      'n_mels': 300,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    return spectrogram