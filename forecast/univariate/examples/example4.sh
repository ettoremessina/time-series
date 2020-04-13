#!/bin/bash
rm -rf dumps/example4
rm -rf logs/example4
rm -rf snaps/example4
rm -rf media/example4_diag

SL=70

python ../uts_gen.py  --tsout timeseries/example4_train.csv --ft "(t / 50.0) + np.sin(t/10.0)" --rend 200

python ../uts_fit.py \
     --traints timeseries/example4_train.csv \
     --samplelength $SL \
     --modelout models/example4 \
     --epochs 100 \
     --batch_size 50 \
     --hlayers 200 300 400 300 200 \
     --hbidirectionals t t t t t \
     --hactivations "tanh" "tanh" "tanh" "tanh" "tanh" \
     --optimizer 'Adamax()'

python ../uts_forecast.py \
    --ts timeseries/example4_train.csv \
    --samplelength $SL \
    --forecastlength 100 \
    --model models/example4 \
    --forecastout forecasts/example4_forecast.csv

#python ../uts_scatter.py --ts timeseries/example4_train.csv --forecast forecasts/example4_forecast.csv
python ../uts_scatter.py --ts timeseries/example4_train.csv --forecast forecasts/example4_forecast.csv --savefig media/example4.png

#python ../uts_diag.py --dump dumps/example4
#python ../uts_diag.py --dump dumps/example4 --savefigdir media/example4_diag
