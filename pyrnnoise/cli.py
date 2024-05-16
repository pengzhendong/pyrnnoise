import click
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from pyrnnoise import RNNoise


@click.command()
@click.argument("in_wav", type=click.Path(exists=True, file_okay=True))
@click.argument("out_wav", type=click.Path(file_okay=True))
@click.option("--plot/--no-plot", default=False, help="Plot the vad probabilities")
def main(in_wav, out_wav, plot):
    info = sf.info(in_wav)
    sr = info.samplerate
    channels = info.channels
    denoiser = RNNoise(sr, channels)
    speech_probs = np.concatenate(list(denoiser.process_wav(in_wav, out_wav)))

    if plot:
        wav, sr = sf.read(in_wav)
        if channels == 1:
            wav = np.expand_dims(wav, axis=1)
        x1 = np.arange(0, len(wav)) / sr
        x2 = [i / 100 for i in range(0, len(speech_probs))]
        for i in range(channels):
            plt.subplot(channels, 1, i + 1)
            plt.plot(x1, wav[:, i])
            plt.plot(x2, speech_probs[:, i])
        plt.show()


if __name__ == "__main__":
    main()
