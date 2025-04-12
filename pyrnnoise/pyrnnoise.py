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

from functools import partial

import numpy as np
from audiolab import AudioGraph, Reader, Writer, filters
from tqdm import tqdm

from .librnnoise import FRAME_SIZE, FRAME_SIZE_MS, SAMPLE_RATE, create, destroy, process_frame


class RNNoise:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.channels = None
        self.denoise_states = None
        self.format = None
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
            aformat = partial(filters.aformat, sample_fmts="s16", channel_layouts=self.layout)
            self._in_graph = AudioGraph(
                rate=self.sample_rate,
                format=self.format,
                layout=self.layout,
                filters=[aformat(sample_rates=SAMPLE_RATE)],
                frame_size=FRAME_SIZE,
            )
        return self._in_graph

    @property
    def out_graph(self):
        if self._out_graph is None:
            aformat = partial(filters.aformat, sample_fmts="s16", channel_layouts=self.layout)
            self._out_graph = AudioGraph(
                rate=SAMPLE_RATE,
                format="s16",
                layout=self.layout,
                filters=[aformat(sample_rates=self.sample_rate)],
            )
        return self._out_graph

    def reset(self):
        self.denoise_states = None
        self._in_graph = None
        if self.sample_rate != SAMPLE_RATE:
            self._out_graph = None

    def process_frame(self, frame: np.ndarray, partial: bool = False):
        if self.denoise_states is None:
            self.denoise_states = [create() for _ in range(self.channels)]
        denoised_frame, speech_probs = process_frame(self.denoise_states, frame)
        if self.sample_rate != SAMPLE_RATE:
            self.out_graph.push(denoised_frame)
            denoised_frame = np.concatenate([frame for frame, _ in self.out_graph.pull(partial)], axis=1)
        return speech_probs, denoised_frame

    def process_chunk(self, chunk: np.ndarray, partial: bool = False):
        chunk = np.atleast_2d(chunk)
        # [num_channels, num_samples]
        self.channels = chunk.shape[0]
        assert chunk.dtype in (np.float32, np.float64, np.int16)
        self.format = "s16" if chunk.dtype == np.int16 else "flt"
        self.in_graph.push(chunk)
        frames = [frame for frame, _ in self.in_graph.pull(partial)]
        for idx, frame in enumerate(frames):
            yield self.process_frame(frame, partial and (idx == len(frames) - 1))

    def process_wav(self, in_path, out_path):
        reader = Reader(in_path, frame_size_ms=FRAME_SIZE_MS)
        writer = Writer(out_path, reader.codec, reader.rate)
        progress_bar = tqdm(
            total=reader.num_frames,
            desc="Denoising",
            unit="frames",
            bar_format="{l_bar}{bar}{r_bar} | {percentage:.2f}%",
        )

        for idx, (frame, _) in enumerate(reader):
            partial = idx == reader.num_frames - 1
            for speech_prob, frame in self.process_chunk(frame, partial):
                writer.write(frame)
                progress_bar.update(1)
                yield speech_prob
        writer.close()
