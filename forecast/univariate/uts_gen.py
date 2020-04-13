import argparse
import numpy as np
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='uts_gen.py generates an univariate an time series')

    parser.add_argument('--tsout',
                        type=str,
                        dest='ts_output_filename',
                        required=True,
                        help='time series output file (csv format)')

    parser.add_argument('--ft',
                        type=str,
                        dest='func_t_body',
                        required=True,
                        help='f(t) body (lamba format)')

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

    #TODO: add arguments for noise

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    t_values = np.arange(args.range_begin, args.range_end, args.range_step, dtype=float)
    func_t = eval('lambda t: ' + args.func_t_body)
    csv_ts_output_file = open(args.ts_output_filename, 'w')
    with csv_ts_output_file:
        writer = csv.writer(csv_ts_output_file, delimiter=',')
        writer.writerow(['y'])
        for i in range(0, t_values.size):
            writer.writerow([func_t(t_values[i])])

    print("#### Terminated {} ####".format(__file__));
