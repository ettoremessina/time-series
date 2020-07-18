#!/bin/bash
EXM=example1_cnn
SL=6
FT="2.0 * np.sin(t/10.0)"
FL=200

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uts_gen.py  \
     --tsout timeseries/${EXM}_train.csv \
     --funct "$FT" \
     --rend 200

 python ../../../../common/uts_gen.py  \
     --tsout timeseries/${EXM}_actual.csv \
     --funct "$FT" \
     --rbegin 200 \
     --rend 400

python ../uts_fit.py \
     --tstrain timeseries/${EXM}_train.csv \
     --samplelength $SL \
     --modelout models/${EXM} \
     --cnnlayers "conv(64, 3, 'relu')" "maxpool(2)" "conv(64, 2, 'tanh')" "maxpool(1)" \
     --epochs 120 \
     --batchsize 40 \
     --optimizer "Adam()" \
     --loss "MeanSquaredError()"
#     --cnnlayers "conv(64, 3, 'relu', 'RandomUniform(minval=-0.1, maxval=0.1)', 'Ones()')" "maxpool(2)" "conv(64, 2, 'tanh')" "maxpool (1)" \
#     --denselayers "dense(80, 'tanh')" \
#     --lstmlayers "lstm(120, 'tanh')" \
#     --metrics "mean_absolute_error" "mean_squared_logarithmic_error" \
#     --dumpout dumps/${EXM} \
#     --logsout logs/${EXM}
#     --modelsnapout snaps/${EXM} \
#     --modelsnapfreq 10

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
    --title "Example #1 by CNN" \
    --xlabel "t" \
    --ylabel "y"

#python ../../common/uts_diagnostic.py --dump dumps/${EXM}
#python ../../common/uts_diagnostic.py --dump dumps/${EXM} --savefigdir media/e${EXM}_diagnostic

#python ../../common/uts_video.py \
#  --modelsnap snaps/${EXM} \
#  --tstrain timeseries/${EXM}_train.csv \
#  --tsactual timeseries/${EXM}_actual.csv \
#  --strategy walk_forward \
#  --samplelength $SL \
#  --fclength $FL \
#  --savevideo media/${EXM}_video.gif \
#  --title "Example #1 by MLP" \
#  --xlabel "t" \
#  --ylabel "y"
