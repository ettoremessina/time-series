#!/bin/bash
rm -rf dumps/example1
rm -rf logs/example1
rm -rf snaps/example1
rm -rf media/example1_diag

SL=5

python ../uts_gen.py  --tsout timeseries/example1_train.csv --ft "2.0 * np.sin(t/10.0)" --rend 200

python ../uts_fit.py \
     --traints timeseries/example1_train.csv \
     --samplelength $SL \
     --modelout models/example1 \
     --epochs 30 \
     --batch_size 50 \
     --hlayers 100 100 \
     --hbidirectionals True True \
     --hactivations 'tanh' 'tanh' \
     --optimizer 'Adam(learning_rate=1e-2, epsilon=1e-07)' \
     --loss 'MeanSquaredError()' \
     --metrics "mean_absolute_error" "mean_squared_logarithmic_error" \
     --dumpout dumps/example1 \
     --logsout logs/example1 \
     --modelsnapout snaps/example1 \
     --modelsnapfreq 5

##python ../fx_fit.py \
##  --trainds datasets/example1_train.csv \
##  --valds datasets/example1_val.csv \
##  --modelout models/example1 \
##  --hlayers 120 160 --hactivations tanh relu \
##  --epochs 100 --batch_size 50 \
##  --optimizer 'Adam(learning_rate=1e-2, epsilon=1e-07)' \
##  --loss 'MeanSquaredError()' \
##  --metrics 'mean_absolute_error' 'mean_squared_logarithmic_error' 'cosine_similarity' \
##  --dumpout dumps/example1 \
##  --logsout logs/example1 \
##  --modelsnapout snaps/example1 \
##  --modelsnapfreq 10

python ../uts_forecast.py \
    --ts timeseries/example1_train.csv \
    --samplelength $SL \
    --forecastlength 200 \
    --model models/example1 \
    --forecastout forecasts/example1_forecast.csv

python ../uts_scatter.py --ts timeseries/example1_train.csv --forecast forecasts/example1_forecast.csv
#python ../uts_scatter.py --ts timeseries/example1_train.csv --forecast forecasts/example1_forecast.csv --savefig media/example1.png

#python ../uts_diag.py --dump dumps/example1
python ../uts_diag.py --dump dumps/example1 --savefigdir media/example1_diag

python ../uts_video.py \
  --modelsnap snaps/example1 \
  --ts timeseries/example1_train.csv \
  --samplelength $SL \
  --forecastlength 200 \
  --savevideo media/example1_video.gif
