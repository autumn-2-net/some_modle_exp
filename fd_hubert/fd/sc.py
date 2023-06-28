from typing import Union

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler



class V3LSGDRLR(_LRScheduler):
    """The WarmupLR schedulerA
        This scheduler is almost same as NoamLR Scheduler except for following
        difference:
        NoamLR:
            lr = optimizer.lr * model_size ** -0.5
                 * min(step ** -0.5, step * warmup_step ** -1.5)
        WarmupLR:
            lr = optimizer.lr * warmup_step ** 0.5
                 * min(step ** -0.5, step * warmup_step ** -1.5)
        Note that the maximum lr equals to optimizer.lr in this scheduler.
        """
    def __init__(self,optimizer: torch.optim.Optimizer,warmup_steps: Union[int, float] = 25000,min_lr=1e-5,last_epoch: int = -1, T_0=1500, eta_max=0.1, eta_min=0., T_mul=2, T_mult=0.9999):
            # assert check_argument_types()
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.eta_min = eta_min
        self.T_0 = T_0
        self.eta_max = eta_max
        self.T_mul = T_mul
        self.T_mult = T_mult
        super().__init__(optimizer, last_epoch)
    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, lr={self.base_lr}, min_lr={self.min_lr}, last_epoch={self.last_epoch})"
    def ctxadjust_lr(self,T_0=10000,eta_min=0.0001,eta_max=2e-4,tmctx=0.96,ws=200):
        step_num = self.last_epoch + 1 +75120 +169020#+360000
        T_cur = (step_num - ws) % T_0
        T_i = T_0
        T_curX = (step_num - ws) // T_0
        cur_lr = eta_min * (tmctx ** T_curX) + 0.5 * (eta_max * (tmctx ** T_curX) - eta_min * (tmctx ** T_curX)) * (
                    1 + np.cos(np.pi * T_cur / T_i))
        if ws > step_num:
            cur_lr = step_num * (eta_max / ws)
        return cur_lr



    def get_lr(self):
        # step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            lrs = []
            for lr in self.base_lrs:
                lr = self.ctxadjust_lr()
                lrs.append(lr)
            return lrs
        else:
            lrs = []
            for lr in self.base_lrs:
                lr = self.ctxadjust_lr()
                lrs.append(lr)
            return lrs

    def set_step(self, step: int):
        self.last_epoch = step
