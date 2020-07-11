import argparse, textwrap
import csv
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=textwrap.dedent('''\
        %(prog)s shows two joined x/y scatter graphs:
        \tthe blue one is the time series
        \tthe red one is the forecast
        \tthe optional orange one is the actual time series
     '''))

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--ts',
                        type=str,
                        dest='timeseries_filename',
                        required=True,
                        help='time series file (csv format)')

    parser.add_argument('--forecast',
                        type=str,
                        dest='forecast_filename',
                        required=True,
                        help='forecast file (csv format)')

    parser.add_argument('--actual',
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

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the chart is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

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
                plt.scatter(t, float(row[0]), color='orange', s=1, marker='.')
                t += 1

    plt.title(args.figure_title)
    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));
