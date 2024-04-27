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
    denoiser = RNNoise(sf.info(in_wav).samplerate)
    vad_probs = list(denoiser.process_wav(in_wav, out_wav))

    if plot:
        wav, sr = sf.read(in_wav)
        x1 = np.arange(0, len(wav)) / sr
        x2 = [i / 100 for i in range(0, len(vad_probs))]
        plt.plot(x1, wav)
        plt.plot(x2, vad_probs)
        plt.show()


if __name__ == "__main__":
    main()
