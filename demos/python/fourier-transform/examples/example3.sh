toolpath="../../../../tools/audio"
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 500.0 -a 32767 -w tmp/example3a.wav
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 1500.0 -a 32767 -w tmp/example3b.wav
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 3500.0 -a 32767 -w tmp/example3c.wav
python $toolpath/wav_16bit_pcm_mono_simplegen.py -d 1.0 -f 4500.0 -a 32767 -w tmp/example3d.wav
python $toolpath/wav_16bit_pcm_mono_combine.py -s1 tmp/example3a.wav -s2 tmp/example3b.wav -w tmp/example3ab.wav
python $toolpath/wav_16bit_pcm_mono_combine.py -s1 tmp/example3c.wav -s2 tmp/example3d.wav -w tmp/example3cd.wav
python $toolpath/wav_16bit_pcm_mono_combine.py -s1 tmp/example3ab.wav -s2 tmp/example3cd.wav -w waves/example3.wav
python $toolpath/wav_spectrogram.py -s waves/example3.wav --savefig spectrograms/spectrogram3.png
