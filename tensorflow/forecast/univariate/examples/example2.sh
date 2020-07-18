#!/bin/bash
rm -rf dumps/example2
rm -rf logs/example2
rm -rf snaps/example2
rm -rf media/example2_diag

SL=6
FT="2.0 * np.sin(t/3.0) / np.exp(t/70)"
FL=200

python ../../../../../common/uts_gen.py  \
     --tsout timeseries/example2_train.csv \
     --funct "$FT" \
     --rend 200

 python ../../../../../common/uts_gen.py  \
     --tsout timeseries/example2_actual.csv \
     --funct "$FT" \
     --rbegin 200 \
     --rend 400

python ../uts_lstm_fit.py \
     --tstrain timeseries/example2_train.csv \
     --samplelength $SL \
     --modelout models/example2 \
     --cnnlayers "conv(128, 3, 'relu')" "maxpool(1)" \
     --lstmlayers "lstm(120, 'tanh')" \
     --epochs 120 \
     --batchsize 50 \
     --optimizer "Adam(learning_rate=1e-3, epsilon=1e-07)" \
     --loss "MeanSquaredError()"
#     --metrics "mean_absolute_error" "mean_squared_logarithmic_error" \
#     --dumpout dumps/example2 \
#     --logsout logs/example2 \
#     --modelsnapout snaps/example2 \
#     --modelsnapfreq 5

python ../../common/uts_forecast.py \
    --tstrain timeseries/example2_train.csv \
    --tsactual timeseries/example2_actual.csv \
    --strategy recursive \
    --samplelength $SL \
    --fclength $FL \
    --model models/example2 \
    --fcout forecasts/example2_forecast.csv \
    --error "MeanSquaredError()"

python ../../../../../common/uts_scatter.py \
    --tstrain timeseries/example2_train.csv \
    --tsforecast forecasts/example2_forecast.csv \
    --tsactual timeseries/example2_actual.csv

#python ../../common/uts_diagnostic.py --dump dumps/example2
#python ../../common/uts_diagnostic.py --dump dumps/example2 --savefigdir media/example2_diag

#python ../../common/uts_video.py \
#  --modelsnap snaps/example2 \
#  --tstrain timeseries/example2_train.csv \
#  --samplelength $SL \
#  --forecastlength $FL \
#  --savevideo media/example2_video.gif
