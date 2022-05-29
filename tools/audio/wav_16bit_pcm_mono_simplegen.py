import argparse
import numpy as np
from scipy.io.wavfile import write

def generate_wav_file():
    t = np.linspace(0., args.duration, int(args.sample_rate * args.duration))
    data = args.amplitude * np.sin(2. * np.pi * args.frequency * t)
    write(args.wav_file_path, args.sample_rate, data.astype(np.int16))

def validate_args():
    #TODO: valid parameters
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s generates a simple and 16 bit PCM mono wav file in according with command line options')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('-d',
                        type=float,
                        dest='duration',
                        required=True,
                        help='duration in seconds')

    parser.add_argument('-f',
                        type=float,
                        dest='frequency',
                        required=True,
                        help='audible frequency in Hz from 20 Hz to 22050 Hz')

    parser.add_argument('-a',
                        type=float,
                        dest='amplitude',
                        required=True,
                        help='amplitude from 0 to 32767')

    parser.add_argument('-w',
                        type=str,
                        dest='wav_file_path',
                        required=True,
                        help='path of the wav file to write')

    parser.add_argument('-r',
                        type=int,
                        dest='sample_rate',
                        required=False,
                        default=44100,
                        help='number of samples per second')

    args = parser.parse_args()
    validate_args()

    generate_wav_file()
