import argparse, textwrap
import csv
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=textwrap.dedent('''\
        %(prog)s shows two joined x/y scatter graphs:
        \tthe blue one is the time series
        \tthe red one is the forecast
        \tthe optional green one is the actual time series
     '''))

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--tstrain',
                        type=str,
                        dest='timeseries_filename',
                        required=True,
                        help='time series file (csv format)')

    parser.add_argument('--tsforecast',
                        type=str,
                        dest='forecast_filename',
                        required=True,
                        help='forecast file (csv format)')

    parser.add_argument('--tsactual',
                        type=str,
                        dest='actual_filename',
                        required=False,
                        help='actual file (csv format)')

    parser.add_argument('--title',
                        type=str,
                        dest='figure_title',
                        required=False,
                        default='',
                        help='if present, it set the title of chart')

    parser.add_argument('--xlabel',
                        type=str,
                        dest='x_axis_label',
                        required=False,
                        default='',
                        help='label of x axis')

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
                        default=9,
                        help='label font size')

    parser.add_argument('--width',
                        type=float,
                        dest='width',
                        required=False,
                        default=9.60,
                        help='width of animated git (in inch)')

    parser.add_argument('--height',
                        type=float,
                        dest='height',
                        required=False,
                        default=5.40,
                        help='height of animated git (in inch)')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the chart is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    plt.rcParams.update({'font.size': args.label_font_size})
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    ax.set_title(args.figure_title, fontdict={'size': args.label_font_size, 'color': 'orange'})
    ax.set_xlabel(args.x_axis_label, fontdict={'size': args.label_font_size})
    ax.set_ylabel(args.y_axis_label, fontdict={'size': args.label_font_size})

    t = 0.0
    with open(args.timeseries_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            plt.scatter(t, float(row[0]), color='blue', s=1, marker='.')
            t += 1
    future0 = t;

    with open(args.forecast_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            plt.scatter(t, float(row[0]), color='red', s=2, marker='.')
            t += 1

    if args.actual_filename:
        t = future0
        with open(args.actual_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                plt.scatter(t, float(row[0]), color='green', s=1, marker='.')
                t += 1

    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));
