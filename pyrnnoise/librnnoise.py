# Copyright (c) 2025, Zhendong Peng (pzd17@tsinghua.org.cn)
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
from typing import List, Union

import numpy as np

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
FRAME_SIZE = lib.rnnoise_get_frame_size()
SAMPLE_RATE = 48000
FRAME_SIZE_MS = FRAME_SIZE * 1000 // SAMPLE_RATE
DTYPE = np.int16


def create() -> ctypes.c_void_p:
    return lib.rnnoise_create(None)


def destroy(state: ctypes.c_void_p):
    lib.rnnoise_destroy(state)


def process_mono_frame(state: ctypes.c_void_p, frame: np.ndarray) -> tuple[np.ndarray, ctypes.c_float]:
    if frame.dtype in (np.float32, np.float64) and -1 <= frame.all() <= 1:
        frame = (frame * 32767).astype(DTYPE)
    assert frame.dtype == DTYPE
    frame = frame.astype(np.float32)

    frame_size = len(frame)
    if frame_size < FRAME_SIZE:
        frame = np.pad(frame, (0, FRAME_SIZE - frame_size))
    ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    speech_prob = lib.rnnoise_process_frame(state, ptr, ptr)
    return frame.astype(DTYPE)[:frame_size], speech_prob


def process_frame(
    states: Union[ctypes.c_void_p, List[ctypes.c_void_p]], frame: np.ndarray
) -> tuple[np.ndarray, ctypes.c_float]:
    if frame.ndim == 1:
        return process_mono_frame(states, frame)
    else:
        # [num_channels, num_samples]
        assert frame.ndim == 2
        assert len(states) == frame.shape[0]
        processed = [process_mono_frame(state, mono_frame) for state, mono_frame in zip(states, frame)]
    frames, speech_probs = zip(*processed)
    return np.row_stack(frames), np.row_stack(speech_probs)
