# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import numpy as np
import soxr


class FrameQueue:
    def __init__(self, frame_size_samples, sample_rate):
        self.frame_size_samples = frame_size_samples
        self.remained_samples = np.empty(0, dtype=np.float32)
        if sample_rate != 48000:
            self.rs = soxr.ResampleStream(sample_rate, 48000, 1, dtype=np.int16)
        self.sample_rate = sample_rate

    def add_chunk(self, chunk, last=False):
        if self.sample_rate != 48000:
            chunk = self.rs.resample_chunk(chunk)
        self.remained_samples = np.concatenate((self.remained_samples, chunk))
        while len(self.remained_samples) >= self.frame_size_samples:
            frame = self.remained_samples[: self.frame_size_samples]
            self.remained_samples = self.remained_samples[self.frame_size_samples :]
            yield frame

        if last and len(self.remained_samples) > 0:
            frame = self.remained_samples
            frame = np.pad(frame, (0, self.frame_size_samples - len(frame)))
            yield frame
