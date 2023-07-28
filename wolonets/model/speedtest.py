import time

import torch

torch.cuda.synchronize()

LLL=torch.nn.Linear(1024,1024).cuda()
ccc=torch.nn.Conv1d(1024,1024,kernel_size=1).cuda()

tttttttttttt=torch.randn(10,1024,200).cuda()

# ttt3 = time.time()
# for i in range(50):
#     LLL(tttttttttttt.transpose(1,2)).transpose(1,2)
# ttt4 = time.time()
#
# # time.sleep(20)
#
# ttt1=time.time()
# for i in range(50):
#     ccc(tttttttttttt)
# ttt2=time.time()

def gettime():
    ttt3 = time.time()

    LLL(tttttttttttt.transpose(1, 2)).transpose(1, 2)
    ttt4 = time.time()

    # time.sleep(20)

    ttt1 = time.time()

    ccc(tttttttttttt)
    ttt2 = time.time()

    return ttt2-ttt1,ttt4-ttt3

ct=0
lt=0
for i in range(100):
    x,y=gettime()
    ct+=x
    lt+=y







print('Conv1d',ct,'Linear:',lt)





















