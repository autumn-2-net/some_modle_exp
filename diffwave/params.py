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


srr = 44100
params = AttrDict(
    # Training params
    batch_size=10,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=srr,
    n_mels=128,
    n_fft=2048,
    hop_samples=512,
    crop_mel_frames=62,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    num_cpu=2,
    drop_last=True,  # 丢批
    pin_memory=True,  # 报仇内存

    # unconditional sample len
    audio_len=srr * 5,  # unconditional_synthesis_samples

    # 优化参数
    interval='epoch',  # 调度的单位，epoch或step
    lrcc=0.9,  # 酸碱率
    lrcl=[10, 20, 30],  # 衰减间隔
    frequency=1  # 衰减器 频率

)
