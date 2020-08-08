#!/bin/bash
EXM=example4_convlstm_dense
SL=20
SSL=5
FT="1.6 * np.sin(t/10.0) + 1.2 * np.cos(t/20.0)"
FL=200

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uvests_gen.py  \
     --tsout timeseries/${EXM}_train.csv \
     --funct "$FT" \
     --tbegin 0 \
     --tend 200 \
     --noise "0.1 * np.random.normal(0, 1, sz)"

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
     --convlstmlayers "convlstm(100, 4, 'tanh')" \
     --denselayers "dense(120, 'tanh')" "dense(120, 'tanh')" \
     --epochs 150 \
     --batchsize 50 \
     --optimizer "Adam()" \
     --loss "MeanSquaredError()"
#     --metrics "mean_absolute_error" "mean_squared_error" \
#     --dumpout dumps/${EXM} \
#     --logsout logs/${EXM} \
#     --modelsnapout snaps/${EXM} \
#     --modelsnapfreq 5 \


python ../fc_uvests_predict.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --strategy recursive \
    --samplelength $SL \
    --subsamplelength $SSL \
    --fclength $FL \
    --model models/${EXM} \
    --tsforecastout forecasts/${EXM}_forecast.csv \
    --error "MeanSquaredError()"

python ../../../../common/fc_uvests_scatter.py \
    --tstrain timeseries/${EXM}_train.csv \
    --tsforecast forecasts/${EXM}_forecast.csv \
    --tsactual timeseries/${EXM}_actual.csv \
    --title "Example #4 by ConvLSTM + Dense" \
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
#  --title "Example #4 by ConvLSTM + Dense" \
#  --tlabel "t" \
#  --ylabel "y"
