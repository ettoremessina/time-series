import argparse
import scipy.io.wavfile as wf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s combine generate the spectrogram of a wav file')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('-s',
                        type=str,
                        dest='source_file_path',
                        required=True,
                        help='wav input file to analyze')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the spectrogram is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    sample_rate, data = wf.read(args.source_file_path)
    channels = data.shape[1] if len(data.shape) > 1 else 1
    data_type = data.dtype
    frames = data.shape[0]
    duration = float(frames) / sample_rate

    print('Channels:\t', channels)
    print('Data type:\t', data_type)
    print('Frame rate:\t', sample_rate)
    print('Frames:\t\t', frames)
    print('Duration:\t', duration, 's')

    max_amp = np.iinfo(data_type).max if np.issubdtype(data_type, np.integer) else 1.0
    samples = data / max_amp

    if (channels > 1):
        monoify = np.zeros(len(samples), dtype=np.float32)
        for channel in range(0, channels):
            monoify = np.add(monoify, samples[:, channel].astype(np.float32))
    else:
        monoify = samples.astype(np.float32)

    times = np.arange(len(data))/float(sample_rate)
    spec_freqs, spec_times, spectrogram = stft(monoify, sample_rate)
    spectrogran = np.abs(spectrogram)

    plt.title('STFT spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.imshow(spectrogran, aspect='auto', origin='lower',
               extent=[spec_times.min(), spec_times.max(), spec_freqs.min(), spec_freqs.max()])

    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()
