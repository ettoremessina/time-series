import argparse
import csv
import time
import numpy as np
import tensorflow.keras.models as tfm

features = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='uts_forecast.py compute forecasting of an univariate time series')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model path')

    parser.add_argument('--ts',
                        type=str,
                        dest='timeserie_filename',
                        required=True,
                        help='time serie file (csv format)')

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

    parser.add_argument('--forecastout',
                        type=str,
                        dest='forecast_data_filename',
                        required=True,
                        help='forecast data file (csv format)')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    y_timeseries = []
    with open(args.timeserie_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            y_timeseries.append(float(row[0]))

    model = tfm.load_model(args.model_path)
    start_time = time.time()

    y_forecast = np.array([])
    to_predict_flat = np.array(y_timeseries[-args.sample_length:])
    for i in range(args.forecast_length):
        to_predict = to_predict_flat.reshape((1, args.sample_length, features))
        ###to_predict = to_predict_flat.reshape((1, 1, args.sample_length, features))
        prediction = model.predict(to_predict, verbose=1)
        print('Forecast of #{}: {}'.format(i+1, prediction))
        y_forecast = np.append(y_forecast, prediction)
        to_predict_flat = np.delete(to_predict_flat, 0)
        to_predict_flat = np.append(to_predict_flat, prediction)

    elapsed_time = time.time() - start_time
    print ("Forecasting time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    csv_output_file = open(args.forecast_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        writer.writerow(['y'])
        for i in range(0, len(y_forecast)):
            writer.writerow([y_forecast[i]])

    print("#### Terminated {} ####".format(__file__));
