import argparse
import csv
import time
import numpy as np
import tensorflow.keras.models as tfm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mpts_mlp_shmo_forecast.py compute forecasting of an univariate time series')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model path')

    parser.add_argument('--tslist',
                        type=str,
                        nargs = '+',
                        dest='timeseries_filenames',
                        required=True,
                        help='parallel time series file list (csv format)')

    parser.add_argument('--samplelength',
                        type=int,
                        dest='sample_length',
                        required=False,
                        default=5,
                        help='sample length')

    parser.add_argument('--forecastlength',
                        type=int,
                        dest='forecast_length',
                        required=False,
                        default=10,
                        help='length of forecast (number of values to predict)')

    parser.add_argument('--fclistout',
                        type=str,
                        nargs = '+',
                        dest='forecast_data_filenames',
                        required=True,
                        help='forecast data file list (csv format) to save')

    args = parser.parse_args()

    print("#### Started mpts_mlp_shmo_forecast ####");

    #todo: check for length
    sequences = []
    for timeseries_filename in args.timeseries_filenames:
        sequence = np.array(read_timeseries(timeseries_filename))
        sequence = sequence[-args.sample_length:]
        sequence.shape = (len(sequence), 1)
        sequences.append(sequence)

    n_sequences = len(sequences)
    n_input = args.sample_length * n_sequences

    sequences = np.hstack(tuple(sequences))

    model = tfm.load_model(args.model_path)
    start_time = time.time()

    to_predict_friendly = np.array(sequences)

    y_forecasts = []
    for j in range (0, n_sequences):
        y_forecasts.append([])

    for i in range(args.forecast_length):
        to_predict = to_predict_friendly.reshape((1, n_input))

        prediction = model.predict(to_predict, verbose=1)
        print('Forecast of #{}: {}'.format(i+1, prediction))

        y_forecast = []
        for j in range (0, n_sequences):
            y_pred = prediction[j][0]
            if len(y_pred.shape) > 0:
                y_pred = y_pred[0]
            y_forecast.append(y_pred)
            y_forecasts[j].append(y_pred)
        to_predict_friendly = np.append(to_predict_friendly, np.array(y_forecast).reshape((1, n_sequences)), axis=0)
        to_predict_friendly = np.delete(to_predict_friendly, 0, axis=0)

    elapsed_time = time.time() - start_time
    print ("Forecasting time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    for j in range (0, n_sequences):
        write_timeseries(args.forecast_data_filenames[j], y_forecasts[j])

    print("#### Terminated mpts_mlp_shmo_forecast ####");
