# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

import time

import torch

class StepTimer:
    """Utility class for measuring steps/second."""

    def __init__(self, step=0.0):
        self.last_iteration = step
        self.start()

    def start(self):
        self.last_time = time.time()

    def steps_per_second(self, cur_step, restart=True):
        value = ((float(cur_step) - self.last_iteration) /
                 (time.time() - self.last_time))
        if restart:
            self.start()
            self.last_iteration = float(cur_step)
        return value


def tensor_to_scalar(x):
    if torch.is_tensor(x):
        return x.item()
    return x


