from collections import OrderedDict

import torch

aaaa=torch.load(r"./epoch=24-step=108000.ckpt")
print(aaaa['state_dict'])
mds=OrderedDict()
ddd={}
for i in aaaa['state_dict']:
    print(i)
    if 'noise' in i:
        continue
    ddd[i] =aaaa['state_dict'][i]
    #ddd[i.replace('diffwav.','')]=aaaa['state_dict'][i]
aaaa['state_dict']=ddd
#mds=OrderedDict(ddd)
torch.save(aaaa,'./clod.cpt')