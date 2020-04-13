#!/bin/bash
rm -rf dumps/example3
rm -rf logs/example3
rm -rf snaps/example3
rm -rf media/example3_diag

python ../uts_gen.py  --tsout timeseries/example3_train.csv --ft "2.0 * np.sin(t/10.0) + 1.5 * np.cos(t/20.0)" --rend 200

python ../uts_fit.py \
     --traints timeseries/example3_train.csv \
     --samplelength 20 \
     --modelout models/example3 \
     --epochs 50 \
     --batch_size 25 \
     --hlayers 200 400  200 \
     --hbidirectionals no no no \
     --hactivations "tanh" "tanh" "tanh" \
     --optimizer "Adamax(learning_rate=1e-3, epsilon=1e-08)" \
     --loss "MeanSquaredError()" \
     --metrics "mean_absolute_error" "mean_squared_logarithmic_error" \
     --dumpout dumps/example3

python ../uts_forecast.py \
    --ts timeseries/example3_train.csv \
    --samplelength 20 \
    --forecastlength 300 \
    --model models/example3 \
    --forecastout forecasts/example3_forecast.csv

#python ../uts_scatter.py --ts timeseries/example3_train.csv --forecast forecasts/example3_forecast.csv
python ../uts_scatter.py --ts timeseries/example3_train.csv --forecast forecasts/example3_forecast.csv --savefig media/example3.png

#python ../uts_diag.py --dump dumps/example3
python ../uts_diag.py --dump dumps/example3 --savefigdir media/example3_diag

#python ../uts_video.py --modelsnap snaps/example3 --ts timeseries/example3_test.csv --savevideo media/example3_test.gif
