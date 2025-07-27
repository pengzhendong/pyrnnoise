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
from audiolab import Reader, Writer
from audiolab.av import AudioGraph, aformat
from tqdm import tqdm

from pyrnnoise.rnnoise import FRAME_SIZE, FRAME_SIZE_MS, SAMPLE_RATE, create, destroy, process_frame


class RNNoise:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.channels = None
        self.denoise_states = None
        self.dtype = None
        self._in_graph = None
        self._out_graph = None

    def __del__(self):
        if self.denoise_states is not None:
            for denoise_state in self.denoise_states:
                destroy(denoise_state)

    @property
    def layout(self):
        assert self.channels in (1, 2)
        return "mono" if self.channels == 1 else "stereo"

    @property
    def in_graph(self):
        if self._in_graph is None:
            self._in_graph = AudioGraph(
                rate=self.sample_rate,
                dtype=self.dtype,
                layout=self.layout,
                filters=[aformat(np.int16, SAMPLE_RATE)],
                frame_size=FRAME_SIZE,
            )
        return self._in_graph

    @property
    def out_graph(self):
        if self._out_graph is None:
            self._out_graph = AudioGraph(
                rate=SAMPLE_RATE,
                dtype=np.int16,
                layout=self.layout,
                filters=[aformat(np.int16, self.sample_rate)],
            )
        return self._out_graph

    def reset(self):
        self.denoise_states = None
        self._in_graph = None
        if self.sample_rate != SAMPLE_RATE:
            self._out_graph = None

    def denoise_frame(self, frame: np.ndarray, partial: bool = False):
        if self.denoise_states is None:
            self.denoise_states = [create() for _ in range(self.channels)]
        denoised_frame, speech_probs = process_frame(self.denoise_states, frame)
        if self.sample_rate != SAMPLE_RATE:
            self.out_graph.push(denoised_frame)
            denoised_frame = np.concatenate([frame for frame, _ in self.out_graph.pull(partial)], axis=1)
        return speech_probs, denoised_frame

    def denoise_chunk(self, chunk: np.ndarray, partial: bool = False):
        chunk = np.atleast_2d(chunk)
        # [num_channels, num_samples]
        self.channels = chunk.shape[0]
        self.dtype = chunk.dtype
        self.in_graph.push(chunk)
        frames = [frame for frame, _ in self.in_graph.pull(partial)]
        for idx, frame in enumerate(frames):
            yield self.denoise_frame(frame, partial and (idx == len(frames) - 1))

    def denoise_wav(self, in_path, out_path):
        reader = Reader(in_path, dtype=np.int16, frame_size_ms=FRAME_SIZE_MS)
        writer = Writer(out_path, reader.rate, reader.codec, layout=reader.layout)
        for idx, (frame, _) in tqdm(enumerate(reader), desc="Denoising", total=reader.num_frames, unit="frames"):
            partial = idx == reader.num_frames - 1
            for speech_prob, frame in self.denoise_chunk(frame, partial):
                writer.write(frame)
                yield speech_prob
        writer.close()
