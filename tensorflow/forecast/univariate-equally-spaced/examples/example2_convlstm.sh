#!/bin/bash
EXM=example2_convlstm
SL=12
SSL=3
FT="2.0 * np.sin(t/5.0) / np.exp(t/80.0)"
FL=200

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_train.csv \
     --funct "$FT" \
     --tend 150

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_actual.csv \
     --funct "$FT" \
     --tbegin 150 \
     --tend 350

python ../fc_uvests_fit.py \
     --tstrain timeseries/${EXM}_train.csv \
     --samplelength $SL \
     --subsamplelength $SSL \
     --modelout models/${EXM} \
     --convlstmlayers "convlstm(128, 3, 'tanh')" "dropout(0.1)"\
     --epochs 80 \
     --batchsize 50 \
     --optimizer "Adam()" \
     --loss "MeanSquaredError()"
#     --metrics "mean_squared_error" \
#     --bestmodelmonitor "mean_squared_error"
#     --dumpout dumps/${EXM} \
#     --logsout logs/${EXM} \
#     --modelsnapout snaps/${EXM} \
#     --modelsnapfreq 5

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
    --title "Example #2 by ConvLSTM" \
    --tlabel "t" \
    --ylabel "y" \
    --savefig media/${EXM}.png

#python ../../../../common/nn_diagnostic.py --dump dumps/${EXM} --savefigdir media/${EXM}_diagnostic

#python ../fc_uvests_video.py \
#  --modelsnap snaps/${EXM} \
#  --tstrain timeseries/${EXM}_train.csv \
#  --samplelength $SL \
#  --subsamplelength $SSL \
#  --forecastlength $FL \
#  --savevideo media/${EXM}_video.gif \
#  --title "Example #1 by ConvLSTM" \
#  --tlabel "t" \
#  --ylabel "y"
