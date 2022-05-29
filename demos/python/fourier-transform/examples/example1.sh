toolpath="../../../../tools/audio"
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 3.0 -f 2500.0 -a 32767 -w waves/example1.wav
python $toolpath/wav_spectrogram.py -s waves/example1.wav --savefig spectrograms/spectrogram1.png
