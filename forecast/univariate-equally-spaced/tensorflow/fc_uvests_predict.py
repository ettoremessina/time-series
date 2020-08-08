import argparse
import csv
import time
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as tfm
import tensorflow.keras.losses as tkl
import tensorflow.keras.layers as tfl

def read_timeseries(tsfilename):
    y_values = []
    with open(tsfilename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            y_values.append(float(row[0]))
    return y_values

def write_timeseries(tsoutfilename, y_forecast):
    csv_output_file = open(tsoutfilename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        writer.writerow(['y'])
        for i in range(0, len(y_forecast)):
            writer.writerow([y_forecast[i]])

def build_error():
    exp_error = 'lambda _ : tkl.' + args.error
    return eval(exp_error)(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s compute forecasting of an univariate equally spaced time series')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model directory path')

    parser.add_argument('--tstrain',
                        type=str,
                        dest='timeseries_filename',
                        required=True,
                        help='univariate equally spaced time series file (in csv format) used for training')

    parser.add_argument('--tsactual',
                        type=str,
                        dest='actual_filename',
                        required=False,
                        help='actual univariate equally spaced time series file (in csv format)')

    parser.add_argument('--strategy',
                        type=str,
                        dest='strategy',
                        required=False,
                        default='recursive',
                        choices=['recursive', 'walk_forward'],
                        help='recursive uses previous predictions as input for future predictions, walk_forward uses actual as input for future predictions (default: %(default)s)')

    parser.add_argument('--samplelength',
                        type=int,
                        dest='sample_length',
                        required=False,
                        default=5,
                        help='length of the sample in terms of number of time steps used for training')

    parser.add_argument('--subsamplelength',
                        type=int,
                        dest='sub_sample_length',
                        required=False,
                        default=1,
                        help='length of the sub sample in terms of number of time steps used for training (it must be a divisor of samplelength; used when a ConvLSTM layer is present or when both CNN and LSTM layers are present, otherwise ignored)')

    parser.add_argument('--fclength',
                        type=int,
                        dest='forecast_length',
                        required=False,
                        default=10,
                        help='length of forecast (number of values to predict)')

    parser.add_argument('--tsforecastout',
                        type=str,
                        dest='forecast_data_filename',
                        required=True,
                        help='output forecast data file (in csv format)')

    parser.add_argument('--error',
                        type=str,
                        dest='error',
                        required=False,
                        help='error function name')

    args = parser.parse_args()

    if args.strategy == 'walk_forward' and args.actual_filename == None:
        raise Exception('walk_forward strategy requires an actual time series')

    if args.error != None and args.actual_filename == None:
        raise Exception('error function requires an actual time series')

    print("#### Started %s ####" % os.path.basename(__file__));

    y_timeseries = read_timeseries(args.timeseries_filename)

    if args.actual_filename:
        y_actual = read_timeseries(args.actual_filename)
        if len(y_actual) < args.forecast_length:
            raise Exception('actual time series length is not enough: it must be at least equals to forecast length')

    model = tfm.load_model(args.model_path)
    if (len(model.layers) < 3):
            raise Exception('invalid model: a model for this program must have at least 3 layers')

    if type(model.layers[1]).__name__ == 'ConvLSTM2D': #workaround for a TF issue
        model_kind = 'convlstm'
    elif isinstance(model.layers[1], tfl.TimeDistributed):
        model_kind = 'cnn-lstm'
    elif isinstance(model.layers[1], tfl.LSTM) or isinstance(model.layers[1], tfl.Bidirectional):
        model_kind = 'lstm'
    elif isinstance(model.layers[1], tfl.Conv1D):
        model_kind = 'cnn'
    elif isinstance(model.layers[1], tfl.Dense):
        model_kind = 'dense'
    else:
        raise Exception('unsupported kind of model: the 2nd layer for this program can be only ConvLSTM, LSTM, CNN or Dense')

    print ('Kind of network: %s' % model_kind)

    start_time = time.time()
    y_forecast = np.array([])
    to_predict_flat = np.array(y_timeseries[-args.sample_length:])
    for i in range(args.forecast_length):
        if model_kind == 'convlstm':
            to_predict = to_predict_flat.reshape((1, args.sub_sample_length, 1, args.sample_length // args.sub_sample_length, 1))
        elif model_kind == 'cnn-lstm':
            to_predict = to_predict_flat.reshape((1, args.sub_sample_length, args.sample_length // args.sub_sample_length, 1))
        elif model_kind == 'cnn' or model_kind == 'lstm':
            to_predict = to_predict_flat.reshape((1, args.sample_length, 1))
        elif model_kind == 'dense':
            to_predict = to_predict_flat.reshape((1, args.sample_length))
        prediction = model.predict(to_predict, verbose=0)[0]
        y_forecast = np.append(y_forecast, prediction)
        to_predict_flat = np.delete(to_predict_flat, 0)
        if args.strategy == 'walk_forward':
            to_predict_flat = np.append(to_predict_flat, y_actual[i])
        else:
            to_predict_flat = np.append(to_predict_flat, prediction)

    elapsed_time = time.time() - start_time
    print ("Forecasting time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    write_timeseries(args.forecast_data_filename, y_forecast)
    print("Generated forecast file '%s'" % args.forecast_data_filename)

    if args.error != None:
        error_func = build_error()
        error_value = error_func(y_actual[:args.forecast_length], y_forecast)
        print('\'%s\' value: %f' % (error_func.name, error_value))

    print("#### Terminated %s ####" % os.path.basename(__file__));
