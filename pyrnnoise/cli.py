import click
import matplotlib.pyplot as plt
import numpy as np
from audiolab import filters, info, load_audio

from pyrnnoise import RNNoise


@click.command()
@click.argument("in_wav", type=click.Path(exists=True, file_okay=True))
@click.argument("out_wav", type=click.Path(file_okay=True))
@click.option("--plot/--no-plot", default=False, help="Plot the vad probabilities")
def main(in_wav, out_wav, plot):
    denoiser = RNNoise(info(in_wav).rate)
    speech_probs = []
    for speech_prob in denoiser.process_wav(in_wav, out_wav):
        speech_probs.append(speech_prob)
    speech_probs = np.concatenate(speech_probs, axis=1)

    if plot:
        audio, rate = load_audio(in_wav, filters=[filters.aformat(sample_fmts="flt")])
        channels = audio.shape[0]
        x1 = np.arange(0, audio.shape[1]) / rate
        x2 = [i / 100 for i in range(0, speech_probs.shape[1])]
        for i, (audio, speech_prob) in enumerate(zip(audio, speech_probs)):
            plt.subplot(channels, 1, i + 1)
            plt.plot(x1, audio)
            plt.plot(x2, speech_prob)
        plt.show()


if __name__ == "__main__":
    main()
