#!/bin/bash
rm -rf dumps/example1
rm -rf logs/example1
rm -rf snaps/example1
rm -rf media/example1_diag

SL=6
FT="2.0 * np.sin(t/10.0)"
FL=200

python ../../../../../common/uts_gen.py  \
     --tsout timeseries/example1_train.csv \
     --funct "$FT" \
     --rend 200

 python ../../../../../common/uts_gen.py  \
     --tsout timeseries/example1_actual.csv \
     --funct "$FT" \
     --rbegin 200 \
     --rend 400

python ../uts_lstm_fit.py \
     --tstrain timeseries/example1_train.csv \
     --samplelength $SL \
     --modelout models/example1 \
     --lstmlayers "lstm(120, 'tanh')" \
     --epochs 80 \
     --batchsize 40 \
     --optimizer "Adam()" \
     --loss "MeanSquaredError()"
#     --cnnlayers "conv(64, 3, 'relu', 'RandomUniform(minval=-0.1, maxval=0.1)', 'Ones()')" "maxpool(2)" "conv(64, 2, 'tanh')" "maxpool (1)" \
#     --denselayers "dense(80, 'tanh', 'TruncatedNormal(mean=0.0, stddev=0.2)', 'TruncatedNormal(mean=0.0, stddev=0.5)')" \
#     --metrics "mean_absolute_error" "mean_squared_logarithmic_error" \
#     --dumpout dumps/example1 \
#     --logsout logs/example1
#     --modelsnapout snaps/example1 \
#     --modelsnapfreq 10

python ../../common/uts_forecast.py \
    --tstrain timeseries/example1_train.csv \
    --tsactual timeseries/example1_actual.csv \
    --strategy recursive \
    --samplelength $SL \
    --fclength $FL \
    --model models/example1 \
    --modelkind cnn \
    --fcout forecasts/example1_forecast.csv \
    --error "MeanSquaredError()"

python ../../../../../common/uts_scatter.py \
    --tstrain timeseries/example1_train.csv \
    --tsforecast forecasts/example1_forecast.csv \
    --tsactual timeseries/example1_actual.csv \
    --title "blue is train, green is actual, red is forecast" \
    --xlabel "t" \
    --ylabel "y"

#python ../../common/uts_diagnostic.py --dump dumps/example1
#python ../../common/uts_diagnostic.py --dump dumps/example1 --savefigdir media/example1_diag

#python ../../common/uts_video.py \
#  --modelsnap snaps/example1 \
#  --tstrain timeseries/example1_train.csv \
#  --tsactual timeseries/example1_actual.csv \
#  --strategy walk_forward \
#  --samplelength $SL \
#  --fclength $FL \
#  --savevideo media/example1_video.gif \
#  --title "blue is train, green is actual, red is forecast" \
#  --xlabel "t" \
#  --ylabel "y"
