#!/bin/bash
EXM=example3_convlstm
SL=40
SSL=10
FT="t/40.0 + 2.0 * np.cos(t/10.0)"
FL=200

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_train.csv \
     --funct "$FT" \
     --tend 200

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_actual.csv \
     --funct "$FT" \
     --tbegin 200 \
     --tend 400

python ../fc_uvests_fit.py \
     --tstrain timeseries/${EXM}_train.csv \
     --samplelength $SL \
     --subsamplelength $SSL \
     --modelout models/${EXM} \
     --convlstmlayers "convlstm(500, 1, 'tanh')" \
     --epochs 40 \
     --batchsize 50 \
     --optimizer "Adam()" \
     --loss "MeanAbsoluteError()"
#     --cnnlayers "conv(64, 3, 'relu', 'RandomUniform(minval=-0.1, maxval=0.1)', 'Ones()')" "maxpool(2)" "conv(64, 2, 'tanh')" "maxpool (1)" \
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
    --tsforecastout forecasts/${EXM}_forecast.csv \
    --error "MeanAbsoluteError()"

python ../../../../common/fc_uvests_scatter.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsforecast forecasts/${EXM}_forecast.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --title "Example #3 by ConvLSTM" \
    --tlabel "t" \
    --ylabel "y" \
    --savefig media/${EXM}.png

#python ../../../../common/nn_dumps_scatter.py --dump dumps/${EXM} --savefigdir media/${EXM}_diagnostic

#python ../fc_uvests_video.py \
#  --modelsnap snaps/${EXM} \
#  --tstrain timeseries/${EXM}_train.csv \
#  --tsactual timeseries/${EXM}_actual.csv \
#  --strategy recursive \
#  --samplelength $SL \
#  --subsamplelength $SSL \
#  --fclength $FL \
#  --savevideo media/${EXM}_video.gif \
#  --title "Example #1 by ConvLSTM" \
#  --tlabel "t" \
#  --ylabel "y"
