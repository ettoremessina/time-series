toolpath="../../../../tools/audio"
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 2500.0 -a 32767 -w tmp/example2a.wav
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 5000.0 -a 32767 -w tmp/example2b.wav
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 7700.0 -a 32767 -w tmp/example2c.wav
python $toolpath/wav_16bit_pcm_mono_concat.py -s1 tmp/example2a.wav -s2 tmp/example2b.wav -w tmp/example2ab.wav
python $toolpath/wav_16bit_pcm_mono_concat.py -s1 tmp/example2ab.wav -s2 tmp/example2c.wav -w waves/example2.wav
python $toolpath/wav_spectrogram.py -s waves/example2.wav --savefig spectrograms/spectrogram2.png
