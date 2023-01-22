from collections import OrderedDict

import torch

aaaa=torch.load("./default/version_12/checkpoints/epoch=7-step=31623.ckpt")
print(aaaa['state_dict'])
mds=OrderedDict()
ddd={}
for i in aaaa['state_dict']:
    print(i)
    ddd[i.replace('diffwav.','')]=aaaa['state_dict'][i]
mds=OrderedDict(ddd)
torch.save(mds,'./ccc.cpt')