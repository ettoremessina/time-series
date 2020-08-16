import argparse
import os
import csv
import numpy as np
import tensorflow.keras.models as tfm
import imageio
import matplotlib.pyplot as plt

def read_timeseries(filename):
    y_values = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            y_values.append(float(row[0]))
    return y_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s generates an animated git that shows the forecast curve computed on an input univariate equally spaced time series as the epochs change.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--modelsnap',
                        type=str,
                        dest='model_snapshots_path',
                        required=True,
                        help='model snapshots directory (generated by uts_fit.py with option --modelsnapout)')

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

    parser.add_argument('--savevideo',
                        type=str,
                        dest='save_gif_video',
                        required=True,
                        default='',
                        help='the animated .gif file name to generate')

    parser.add_argument('--title',
                        type=str,
                        dest='figure_title_prefix',
                        required=False,
                        default='',
                        help='if present, it set the prefix title of chart')

    parser.add_argument('--tlabel',
                        type=str,
                        dest='t_axis_label',
                        required=False,
                        default='',
                        help='label of t axis')

    parser.add_argument('--ylabel',
                        type=str,
                        dest='y_axis_label',
                        required=False,
                        default='',
                        help='label of y axis')

    parser.add_argument('--labelfontsize',
                        type=int,
                        dest='label_font_size',
                        required=False,
                        default=18,
                        help='label font size')

    parser.add_argument('--frameperseconds',
                        type=int,
                        dest='frame_per_seconds',
                        required=False,
                        default=10,
                        help='frame per seconds')

    parser.add_argument('--width',
                        type=float,
                        dest='width',
                        required=False,
                        default=19.20,
                        help='width of animated git (in inch)')

    parser.add_argument('--height',
                        type=float,
                        dest='height',
                        required=False,
                        default=10.80,
                        help='height of animated git (in inch)')

    args = parser.parse_args()

    if args.strategy == 'walk_forward' and args.actual_filename == None:
        raise Exception('walk_forward strategy requires an actual time series')

    print("#### Started %s ####" % os.path.basename(__file__));

    y_timeseries = read_timeseries(args.timeseries_filename)
    y_actual = []
    if args.actual_filename:
        y_actual = read_timeseries(args.actual_filename)
        if len(y_actual) < args.forecast_length:
            raise Exception('actual time series length is not enough: it must be at least equals to forecast length')

    miny = min(np.concatenate([y_timeseries, y_actual]))
    maxy = max(np.concatenate([y_timeseries, y_actual]))

    miny = miny - 0.1 * (maxy - miny)
    maxy = maxy + 0.1 * (maxy - miny)

    frames = []
    plt.rcParams.update({'font.size': args.label_font_size})
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    epochs = [mdl for mdl in sorted(os.listdir(args.model_snapshots_path))]
    for epoch in epochs:
        model = tfm.load_model(os.path.join(args.model_snapshots_path, epoch))

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

        plt.cla()
        ax.set_xlim(0, len(y_timeseries) + args.forecast_length)
        ax.set_ylim(miny, maxy)

        title_prefix = ''
        if args.figure_title_prefix != '':
            title_prefix = args.figure_title_prefix + ' ';
        ax.set_title(title_prefix + '[Epoch = %d]' % int(epoch), fontdict={'size': args.label_font_size, 'color': 'orange'})

        ax.set_xlabel(args.t_axis_label, fontdict={'size': args.label_font_size})
        ax.set_ylabel(args.y_axis_label, fontdict={'size': args.label_font_size})
        plt.scatter(range(len(y_timeseries)), y_timeseries, color='blue', s=2, marker='.')
        plt.scatter(range(len(y_timeseries), len(y_timeseries) + len(y_actual)), y_actual, color='green', s=2, marker='.')
        plt.scatter(range(len(y_timeseries), len(y_timeseries) + args.forecast_length), y_forecast, color='red', s=2, marker='.')

        # Used to return the plot as an image array
        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame  = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        print ("Generated frame for epoch %s" % epoch)

    imageio.mimsave(args.save_gif_video, frames, fps=args.frame_per_seconds)
    print ("Generated '%s' animated gif" % args.save_gif_video)

    print("#### Terminated %s ####" % os.path.basename(__file__));