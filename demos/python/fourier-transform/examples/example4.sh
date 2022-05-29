toolpath="../../../../tools/audio"
python $toolpath/wav_spectrogram.py -s ./select-mvoice.wav --savefig spectrograms/spectrogram4m.png
python $toolpath/wav_spectrogram.py -s ./select-fvoice.wav --savefig spectrograms/spectrogram4f.png
