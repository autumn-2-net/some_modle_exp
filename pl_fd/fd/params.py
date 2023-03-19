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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


# srr = 44100
params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=0.0002,
    max_grad_norm=None,  # 梯度裁切

    # Data params 预处理参数 及训练
    sample_rate=44100,
    n_mels=128,
    n_fft=2048,
    hop_samples=512,  # 目前只正常256  512
    crop_mel_frames=100,  # Probably an error in paper. 切割片
    val_crop_mel_frames=150,
    pre_power=1.0,
    f_min=40,
    f_max=16000,
    win_length=512 * 4,

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=False,
    noise_schedule=np.linspace(1e-4, 0.05, 1000).tolist(),  # 层
    inference_noise_schedule=[0.00015496854030061513,
                                 0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5],  # 加速
    num_cpu=4,  # dl进程
    drop_last=True,  # 丢批
    pin_memory=True,  # 报仇内存

    # unconditional sample len
    audio_len=44100 * 5,  # unconditional_synthesis_samples 没用不用管

    # 优化参数
    interval='epoch',  # 调度的单位，epoch或step
    lrcc=0.9,  # 酸碱率 衰减
    lrcl=[1, 1, 5, 20, 30],  # 衰减间隔
    frequency=1,  # 衰减器 频率
    valst=2000,  # 验证
    loger='TB',  # TB or wandb

)
