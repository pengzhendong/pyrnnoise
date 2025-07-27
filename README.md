# pyrnnoise

[![PyPI](https://img.shields.io/pypi/v/pyrnnoise)](https://pypi.org/project/pyrnnoise/)
[![License](https://img.shields.io/github/license/pengzhendong/pyrnnoise)](LICENSE)

Python bindings for [RNNoise](https://github.com/xiph/rnnoise), a recurrent neural network for audio noise reduction.

## Features

- Real-time noise suppression for speech audio
- Command-line interface for processing audio files
- Supports mono and stereo audio files
- Visualize voice activity detection probabilities

## Installation

```bash
pip install pyrnnoise
```

## Usage

### Command-line interface

```bash
# Basic usage
denoise input.wav output.wav

# With voice activity detection plot
denoise input.wav output.wav --plot
```

### Python API

```python
from pyrnnoise import RNNoise

# Create denoiser instance
denoiser = RNNoise(sample_rate=48000)

# Process audio file
for speech_prob in denoiser.denoise_wav("input.wav", "output.wav"):
    print(f"Processing frame with speech probability: {speech_prob}")
```

#### Advanced Usage

The `RNNoise` class provides several methods for processing audio at different levels:

- `denoise_frame(frame, partial=False)`: Process a single audio frame (480 samples at 48kHz)
  - Returns a tuple of (speech_probabilities, denoised_frame)
  - speech_probabilities: Voice activity detection probabilities for each channel
  - denoised_frame: The denoised audio frame

- `denoise_chunk(chunk, partial=False)`: Process a chunk of audio data
  - Takes a numpy array of audio samples [num_channels, num_samples]
  - Yields tuples of (speech_probabilities, denoised_frame) for each frame
  - Useful for processing audio streams or large audio files in chunks

Example using denoise_chunk:
```python
import numpy as np
from pyrnnoise import RNNoise

# Create denoiser instance
denoiser = RNNoise(sample_rate=48000)

# Generate or load some audio data (stereo in this example)
audio_data = np.random.randint(-32768, 32767, (2, 48000), dtype=np.int16)

# Process audio chunk
for speech_prob, denoised_audio in denoiser.denoise_chunk(audio_data):
    print(f"Speech probability: {speech_prob}")
    # Process denoised_audio as needed
```

## Build from source

```bash
# Clone with submodules
git submodule update --init

# Build RNNoise library
cmake -B pyrnnoise/build -DCMAKE_BUILD_TYPE=Release
cmake --build pyrnnoise/build --target install

# Install Python package in development mode
pip install -e .
```

## License

[Apache License 2.0](LICENSE)
