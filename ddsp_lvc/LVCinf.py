# from lvc.dataset2 import from_path, from_gtzan
import numpy as np
import torch
import torchaudio

from lvct_gan import PL_diffwav
from lvc.params import params


spectrogram = torch.from_numpy(np.load(r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\test\2100003756.wav.spec.npy')).cuda()

md = PL_diffwav(params)
eee = torch.load(r'C:\Users\autumn\Desktop\poject_all\vcoder\LVC_net\mdscpscxVGX\sample-mnist-epoch25-25-264468.ckpt')[
    'state_dict']
md.load_state_dict(eee,strict=False)
md.cuda().eval()

icxa=spectrogram.chunk(1,dim=1)
wavd=[]
with torch.no_grad():
    for i in icxa:
        t=len(i.t())
        noist = torch.randn(1, 16, t * 512, ).cuda()*500
        aaac=md(noist,i.unsqueeze(0))[0]
        wavd.append(aaac)

wsss=torch.cat(wavd,dim=1)

# ttccc=[]
# conv3 = torch.nn.Conv2d(1, 1, (5, 5),bias=False,padding=2)
# w1 = torch.Tensor(np.array([[1/25, 1/25,1/25,1/25,1/25,],[1/25, 1/25,1/25,1/25,1/25,],[1/25, 1/25,1/25,1/25,1/25,],[1/25, 1/25,1/25,1/25,1/25,],[1/25, 1/25,1/25,1/25,1/25,],]).reshape(1, 1, 5,5))
# conv3.weight = torch.nn.Parameter(w1)
# conv3.cuda()
# with torch.no_grad():
#     for i in range(10):
#         t = len(spectrogram.t())
#         noist = torch.randn(1, 16, t * 512, ).cuda()
#
#         # noist=torch.ones_like(noist).cuda()
#         aaac = md(noist, spectrogram.unsqueeze(0).unsqueeze(0).squeeze(0))[0]
#         ttccc.append(aaac)
# aaasdc=0
# for i in ttccc:
#     aaasdc+=i
# aaasdc=aaasdc/10
#
#
#
#
# torchaudio.save('./o/gs31.wav', aaasdc.cpu(), sample_rate=44100)
torchaudio.save('./o/n3.wav', wsss.cpu(), sample_rate=44100)