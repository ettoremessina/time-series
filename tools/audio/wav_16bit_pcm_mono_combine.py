import argparse
import numpy as np
from scipy.io.wavfile import read, write

def concatenate_wav_files():
    samplerate1, data1 = read(args.source1_file_path)
    data1 = np.array(data1, dtype=np.float64)
    samplerate2, data2 = read(args.source2_file_path)
    data1 = np.array(data1, dtype=np.float64)
    if samplerate1 != samplerate2:
        raise Exception("sample rate mismatch: the two source wav files must have same samplerate")

    l1, l2 = [len(data1), len(data2)]
    if l1 > l2:
        lpad = l1 - l2
        data2 = np.concatenate((data2, np.zeros(lpad)))
    elif l1 < l2:
        lpad = l2 - l1
        data1 = np.concatenate((data1, np.zeros(lpad)))

    data = (data1 + data2) / 2
    write(args.final_file_path, samplerate1, data.astype(np.int16))

def validate_args():
    #TODO: valid parameters
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s combines two 16 bit PCM mono wav files')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('-s1',
                        type=str,
                        dest='source1_file_path',
                        required=True,
                        help='first source wav file')

    parser.add_argument('-s2',
                        type=str,
                        dest='source2_file_path',
                        required=True,
                        help='second source wav file')

    parser.add_argument('-w',
                        type=str,
                        dest='final_file_path',
                        required=True,
                        help='path of the final combined wav file to write')

    args = parser.parse_args()
    validate_args()

    concatenate_wav_files()
