#!/bin/sh

EXM=example1_shmo_dense
SL=10
FAT="2.0 * np.sin(t/10.0)"
FBT="3.0 * np.cos(t/20.0)"
FCT="2.0 * np.sin(t/10.0) + 3 * np.cos(t/20)"
FL=100

rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -rf media/${EXM}_diagnostic

python ../../../../common/uvests_gen.py  \
  --tsout timeseries/${EXM}_train_a.csv \
  --funct "$FAT" \
  --tend 200

python ../../../../common/uvests_gen.py  \
  --tsout timeseries/${EXM}_train_b.csv \
  --funct "$FBT" \
  --tend 200

python ../../../../common/uvests_gen.py \
  --tsout timeseries/${EXM}_train_c.csv \
  --funct "$FCT" \
  --tend 200

python ../fc_mvests_shmo_fit.py \
     --traintslist timeseries/${EXM}_train_a.csv timeseries/${EXM}_train_b.csv timeseries/${EXM}_train_c.csv \
     --samplelength $SL \
     --modelout models/${EXM} \
     --epochs 100 \
     --batch_size 50 \
     --hlayers 50 50 \
     --hactivations "tanh" "tanh" \
     --optimizer "Adam()" \
     --loss "MeanSquaredError()"
#     --metrics "mean_absolute_error" "mean_squared_logarithmic_error" \
#     --dumpout dumps/example1 \
#     --logsout logs/example1 \
#     --modelsnapout snaps/example1 \
#     --modelsnapfreq 5

python ../fc_mvests_shmo_predict.py \
    --tslist timeseries/${EXM}_train_a.csv timeseries/${EXM}_train_b.csv timeseries/${EXM}_train_c.csv \
    --samplelength $SL \
    --forecastlength $FL \
    --model models/${EXM} \
    --fclistout forecasts/${EXM}_forecast_a.csv forecasts/${EXM}_forecast_b.csv forecasts/${EXM}_forecast_c.csv

python ../../../../common/fc_uvests_scatter.py \
  --tstrain timeseries/${EXM}_train_a.csv \
  --tsforecast forecasts/${EXM}_forecast_a.csv

python ../../../../common/fc_uvests_scatter.py \
  --tstrain timeseries/${EXM}_train_b.csv \
  --tsforecast forecasts/${EXM}_forecast_b.csv

python ../../../../common/fc_uvests_scatter.py \
  --tstrain timeseries/${EXM}_train_c.csv \
  --tsforecast forecasts/${EXM}_forecast_c.csv

#python ../uts_diag.py --dump dumps/example1
#python ../uts_diag.py --dump dumps/example1 --savefigdir media/example1_diag

#python ../uts_video.py \
#  --modelsnap snaps/example1 \
#  --ts timeseries/example1_train.csv \
#  --samplelength $SL \
#  --forecastlength 200 \
#  --savevideo media/example1_video.gif
