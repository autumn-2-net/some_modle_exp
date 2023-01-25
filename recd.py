from collections import OrderedDict

import torch

aaaa=torch.load(r"./default/version_51/checkpoints/epoch=14-step=158228.ckpt")
print(aaaa['state_dict'])
mds=OrderedDict()
ddd={}
for i in aaaa['state_dict']:
    print(i)
    ddd[i.replace('diffwav.','')]=aaaa['state_dict'][i]
mds=OrderedDict(ddd)
torch.save(mds,'./ccci34.cpt')