#!/bin/bash
EXM=example1_cnn_lstm
SL=6
SSL=2
FT="2.0 * np.sin(t/10.0)"
FL=400

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_train.csv \
     --funct "$FT" \
     --tbegin 0 \
     --tend 200 \
     --tstep 0.5

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_actual.csv \
     --funct "$FT" \
     --tbegin 200 \
     --tend 400 \
     --tstep 0.5

python ../fc_uvests_fit.py \
     --tstrain timeseries/${EXM}_train.csv \
     --samplelength $SL \
     --subsamplelength $SSL \
     --modelout models/${EXM} \
     --cnnlayers "conv(128, 3, 'relu')" "maxpool(1)" \
     --lstmlayers "lstm(120, 'tanh')" \
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

python ../fc_uvests_predict.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --strategy recursive \
    --samplelength $SL \
    --subsamplelength $SSL \
    --fclength $FL \
    --model models/${EXM} \
    --fcout forecasts/${EXM}_forecast.csv \
    --error "MeanSquaredError()"

python ../../../../common/fc_uvests_scatter.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsforecast forecasts/${EXM}_forecast.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --title "Example #1 by CNN + LSTM" \
    --tlabel "t" \
    --ylabel "y" \
    --savefig media/${EXM}.png

#python ../../../../common/nn_diagnostic.py --dump dumps/${EXM} --savefigdir media/${EXM}_diagnostic

#python ../fc_uvests_video.py \
#  --modelsnap snaps/${EXM} \
#  --tstrain timeseries/${EXM}_train.csv \
#  --tsactual timeseries/${EXM}_actual.csv \
#  --strategy recursive \
#  --samplelength $SL \
#  --subsamplelength $SSL \
#  --fclength $FL \
#  --savevideo media/${EXM}_video.gif \
#  --title "Example #1 by CNN + LSTM" \
#  --tlabel "t" \
#  --ylabel "y"
