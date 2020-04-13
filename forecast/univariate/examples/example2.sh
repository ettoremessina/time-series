#!/bin/bash
rm -rf dumps/example2
rm -rf logs/example2
rm -rf snaps/example2
rm -rf media/example2_diag

python ../uts_gen.py  --tsout timeseries/example2_train.csv --ft "2.0 * np.sin(t/3.0) / np.exp(t/70)" --rend 200

python ../uts_fit.py \
     --traints timeseries/example2_train.csv \
     --samplelength 5 \
     --modelout models/example2 \
     --epochs 30 \
     --batch_size 50 \
     --hlayers 100 100 \
     --hbidirectionals yes no \
     --hactivations 'tanh' 'tanh' \
     --optimizer 'Adam(learning_rate=1e-2, epsilon=1e-07)' \
     --loss 'MeanSquaredError()'

python ../uts_forecast.py \
    --ts timeseries/example2_train.csv \
    --samplelength 5 \
    --forecastlength 200 \
    --model models/example2 \
    --forecastout forecasts/example2_forecast.csv

#python ../uts_scatter.py --ts timeseries/example2_train.csv --forecast forecasts/example2_forecast.csv
python ../uts_scatter.py --ts timeseries/example2_train.csv --forecast forecasts/example2_forecast.csv --savefig media/example2.png

#python ../uts_diag.py --dump dumps/example2
#python ../uts_diag.py --dump dumps/example2 --savefigdir media/example2_diag

#python ../uts_video.py --modelsnap snaps/example2 --ts timeseries/example2_test.csv --savevideo media/example2_test.gif
