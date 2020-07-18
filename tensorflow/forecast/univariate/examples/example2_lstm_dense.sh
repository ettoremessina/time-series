#!/bin/bash
EXM=example2_lstm_dense
SL=12
FT="2.0 * np.sin(t/5.0) / np.exp(t/80.0)"
FL=200

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uts_gen.py  \
     --tsout timeseries/${EXM}_train.csv \
     --funct "$FT" \
     --rend 150

 python ../../../../common/uts_gen.py  \
     --tsout timeseries/${EXM}_actual.csv \
     --funct "$FT" \
     --rbegin 150 \
     --rend 350

python ../uts_fit.py \
     --tstrain timeseries/${EXM}_train.csv \
     --samplelength $SL \
     --modelout models/${EXM} \
     --lstmlayers "lstm(120, 'tanh')" \
     --denselayers "dense(120, 'tanh')" "dense(120, 'tanh')" \
     --epochs 120 \
     --batchsize 50 \
     --optimizer "Adam(learning_rate=1e-3, epsilon=1e-07)" \
     --loss "MeanSquaredError()"
     #--metrics "mean_squared_error" \
     #--bestmodelmonitor "mean_squared_error"
#     --dumpout dumps/${EXM} \
#     --logsout logs/${EXM} \
#     --modelsnapout snaps/${EXM} \
#     --modelsnapfreq 5

python ../uts_forecast.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --strategy recursive \
    --samplelength $SL \
    --fclength $FL \
    --model models/${EXM} \
    --fcout forecasts/${EXM}_forecast.csv \
    --error "MeanSquaredError()"

python ../../../../common/uts_scatter.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsforecast forecasts/${EXM}_forecast.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --title "Example #2 by LSTM + Dense" \
    --xlabel "t" \
    --ylabel "y"

#python ../../common/uts_diagnostic.py --dump dumps/${EXM}
#python ../../common/uts_diagnostic.py --dump dumps/${EXM} --savefigdir media/${EXM}_diagnostic

#python ../../common/uts_video.py \
#  --modelsnap snaps/${EXM} \
#  --tstrain timeseries/${EXM}_train.csv \
#  --samplelength $SL \
#  --forecastlength $FL \
#  --savevideo media/${EXM}_video.gif
