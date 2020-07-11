import argparse
import numpy as np
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s generates an univariate time series')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--tsout',
                        type=str,
                        dest='ts_output_filename',
                        required=True,
                        help='time series output file (csv format)')

    parser.add_argument('--funct',
                        type=str,
                        dest='func_t_body',
                        required=True,
                        help='func(t) body (lamba format)')

    parser.add_argument('--rbegin',
                        type=float,
                        dest='range_begin',
                        required=False,
                        default=0.0,
                        help='begin range (default:0)')

    parser.add_argument('--rend',
                        type=float,
                        dest='range_end',
                        required=False,
                        default=+100.0,
                        help='end range (default:100)')

    parser.add_argument('--rstep',
                        type=float,
                        dest='range_step',
                        required=False,
                        default=1.0,
                        help='step range (default: 1.0)')

    parser.add_argument('--noise',
                        type=str,
                        dest='noise_body',
                        required=False,
                        help='noise(sz) body (lamba format)')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    t_values = np.arange(args.range_begin, args.range_end, args.range_step, dtype=float)
    func_t = eval('lambda t: ' + args.func_t_body)

    if args.noise_body:
        noise_sz = eval('lambda sz: ' + args.noise_body)
        noise =  noise_sz(len(t_values))
    else:
        noise = [0] * len(t_values)

    csv_ts_output_file = open(args.ts_output_filename, 'w')
    with csv_ts_output_file:
        writer = csv.writer(csv_ts_output_file, delimiter=',')
        writer.writerow(['y'])
        for i in range(0, t_values.size):
            writer.writerow([func_t(t_values[i]) + noise[i]])
    print("Generated univariate time series file '%s'" % args.ts_output_filename)

    print("#### Terminated %s ####" % os.path.basename(__file__));
