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
import warnings
import wave

import numpy as np
import soxr
from tqdm import tqdm


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
    def __init__(self):
        self.denoise_state = lib.rnnoise_create(None)
        self.frame_size = lib.rnnoise_get_frame_size()

    def __del__(self):
        lib.rnnoise_destroy(self.denoise_state)

    def process_frame(self, frame):
        if frame.shape[0] != self.frame_size:
            warnings.warn(
                f"The current frame size {frame.shape[0]} is less than {self.frame_size}."
                "If it is the last frame, please pad 0."
            )
        data = frame.astype(np.float32)
        in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vad_prob = lib.rnnoise_process_frame(self.denoise_state, in_ptr, in_ptr)
        return vad_prob, data.astype(np.int16)

    def process_wav(self, in_path, out_path):
        with wave.open(in_path) as in_wav, wave.open(out_path, "w") as out_wav:
            assert in_wav.getnchannels() == 1
            assert in_wav.getsampwidth() == 2
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)

            sr = in_wav.getframerate()
            out_wav.setframerate(sr)
            if sr != 48000:
                in_rs = soxr.ResampleStream(sr, 48000, 1, dtype=np.int16)
                out_rs = soxr.ResampleStream(48000, sr, 1, dtype=np.int16)

            frame_size = sr * self.frame_size // 48000
            n_frames = in_wav.getnframes() // frame_size
            progress_bar = tqdm(total=n_frames, desc="Denoising audio", unit="frames")
            while True:
                frame = in_wav.readframes(frame_size)
                if not frame:
                    break
                frame = np.frombuffer(frame, dtype=np.int16)
                if sr != 48000:
                    frame = in_rs.resample_chunk(frame)
                frame = np.pad(frame, (0, self.frame_size - frame.shape[0]))
                speech_prob, frame = self.process_frame(frame)
                if sr != 48000:
                    frame = out_rs.resample_chunk(frame)
                out_wav.writeframes(frame)
                progress_bar.update(1)
                yield speech_prob
