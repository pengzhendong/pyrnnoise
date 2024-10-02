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

import ctypes
import math
import os
import platform

import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

from .frame_queue import FrameQueue


if platform.system() == "Darwin":
    LIBRNNOISE = os.path.join(os.path.dirname(__file__), "librnnoise.dylib")
elif platform.system() == "Windows":
    LIBRNNOISE = os.path.join(os.path.dirname(__file__), "rnnoise.dll")
elif platform.system() == "Linux":
    LIBRNNOISE = os.path.join(os.path.dirname(__file__), "librnnoise.so")
else:
    raise OSError("Unsupported operating system")
if not os.path.exists(LIBRNNOISE):
    raise OSError(f"RnNoise library not found: {LIBRNNOISE}")
lib = ctypes.CDLL(LIBRNNOISE)
lib.rnnoise_create.argtypes = [ctypes.c_void_p]
lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
lib.rnnoise_process_frame.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.rnnoise_create.restype = ctypes.c_void_p
lib.rnnoise_get_frame_size.restype = ctypes.c_int
lib.rnnoise_process_frame.restype = ctypes.c_float


class RNNoise:
    def __init__(self, sample_rate, channels=1):
        self.denoise_states = [lib.rnnoise_create(None) for _ in range(channels)]
        self.frame_size_samples = lib.rnnoise_get_frame_size()
        self.frame_size_ms = self.frame_size_samples * 1000 // 48000

        self.channels = channels
        self.sample_rate = sample_rate
        if self.sample_rate != 48000:
            self.rs = soxr.ResampleStream(48000, self.sample_rate, self.channels)
        self.queue = FrameQueue(
            self.frame_size_samples, self.sample_rate, self.channels
        )

    def __del__(self):
        for denoise_state in self.denoise_states:
            lib.rnnoise_destroy(denoise_state)

    def reset(self):
        if self.sample_rate != 48000:
            self.rs = soxr.ResampleStream(48000, self.sample_rate, self.channels)
        self.queue = FrameQueue(
            self.frame_size_samples, self.sample_rate, self.channels
        )

    def process_frame(self, frame, last):
        frame_size = len(frame)
        speech_probs = np.empty((1, self.channels), dtype=np.float32)
        denoised_frame = np.empty((frame_size, self.channels), dtype=np.float32)
        if frame_size < self.frame_size_samples:
            frame = np.pad(frame, ((0, self.frame_size_samples - frame_size), (0, 0)))
        for i in range(self.channels):
            state = self.denoise_states[i]
            # scale the frame to the range of int16, but in float32
            data = (frame[:, i] * 32768).astype(np.float32)
            in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            speech_probs[:, i] = lib.rnnoise_process_frame(state, in_ptr, in_ptr)
            # scale the denoised frame back to the range of [-1.0, 1.0]
            denoised_frame[:, i] = data.astype(np.int16)[:frame_size] / 32768
        if self.sample_rate != 48000:
            denoised_frame = self.rs.resample_chunk(denoised_frame, last)
        return speech_probs, denoised_frame

    def process_chunk(self, chunk, last=False):
        # expand dims for the mono audio chunk
        if self.channels == 1 and len(chunk.shape) == 1:
            chunk = np.expand_dims(chunk, axis=1)
        for frame, last in self.queue.add_chunk(chunk, last):
            yield self.process_frame(frame, last)

    def process_wav(self, in_path, out_path, block_size=None):
        info = sf.info(in_path)
        sr = info.samplerate
        channels = info.channels
        subtype = info.subtype
        assert sr == self.sample_rate
        assert channels == self.channels

        progress_bar = tqdm(
            total=math.ceil(info.duration * 1000 / self.frame_size_ms),
            desc="Denoising audio",
            unit="frames",
            bar_format="{l_bar}{bar}{r_bar} | {percentage:.2f}%",
        )
        if block_size is None:
            block_size = 10 * sr // 1000
        blocks = sf.blocks(in_path, blocksize=block_size)

        self.reset()
        with sf.SoundFile(out_path, "w", sr, channels, subtype) as out_wav:
            next_block = next(blocks, None)
            while next_block is not None:
                block = next_block.astype(np.float32)
                next_block = next(blocks, None)
                last = next_block is None
                for speech_prob, frame in self.process_chunk(block, last):
                    out_wav.write(frame)
                    progress_bar.update(1)
                    yield speech_prob
