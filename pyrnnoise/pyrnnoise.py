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
import os
import platform
import wave

import numpy as np
import soxr
from tqdm import tqdm

from .frame_queue import FrameQueue


if platform.system() == "Darwin":
    LIBRNNOISE = os.path.join(os.path.dirname(__file__), "librnnoise.dylib")
elif platform.system() == "Windows":
    LIBRNNOISE = os.path.join(os.path.dirname(__file__), "librnnoise.dll")
elif platform.system() == "Linux":
    LIBRNNOISE = os.path.join(os.path.dirname(__file__), "librnnoise.so")
else:
    raise OSError("Unsupported operating system")
if not os.path.exists(LIBRNNOISE):
    raise OSError(f"RNNoise library not found: {LIBRNNOISE}")
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
    def __init__(self, sample_rate):
        self.denoise_state = lib.rnnoise_create(None)
        self.frame_size_samples = lib.rnnoise_get_frame_size()
        self.queue = FrameQueue(self.frame_size_samples, sample_rate)

    def __del__(self):
        lib.rnnoise_destroy(self.denoise_state)

    def process_chunk(self, chunk):
        for frame in self.queue.add_chunk(chunk):
            data = frame.astype(np.float32)
            in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            vad_prob = lib.rnnoise_process_frame(self.denoise_state, in_ptr, in_ptr)
            yield vad_prob, data.astype(np.int16)

    def process_wav(self, in_path, out_path):
        with wave.open(in_path) as in_wav, wave.open(out_path, "w") as out_wav:
            sr = in_wav.getframerate()
            assert in_wav.getnchannels() == 1
            assert in_wav.getsampwidth() == 2
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(sr)

            # chunk_size_samples = self.frame_size_samples
            chunk_size_samples = 30 * sr // 1000
            n_frames = in_wav.getnframes() // self.frame_size_samples
            progress_bar = tqdm(total=n_frames, desc="Denoising audio", unit="chunks", bar_format="{l_bar}{bar}{r_bar} | {percentage:.2f}%")
            while True:
                chunk = in_wav.readframes(chunk_size_samples)
                if not chunk:
                    break
                chunk = np.frombuffer(chunk, dtype=np.int16)
                for speech_prob, frame in self.process_chunk(chunk):
                    out_wav.writeframes(frame)
                    progress_bar.update(1)
                    yield speech_prob
