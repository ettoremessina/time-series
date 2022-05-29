import argparse
import csv
import time
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as tko
import tensorflow.keras.activations as tka
import tensorflow.keras.losses as tkl
import tensorflow.keras.metrics as tkm
import tensorflow.keras.callbacks as tfcb
import tensorflow.keras.initializers as tfi
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_samples(seqs):
    X, y = [], []
    for i in range(len(seqs)):
        ex = i + args.sample_length
        if ex > len(seqs) - 1:
            break
        sX, sy = seqs[i:ex, :], seqs[ex, :]
        X.append(sX)
        y.append(sy)
    return np.array(X), np.array(y)

def build_model(n_input, n_output):
    inputs = Input(shape=(n_input,))
    hidden = inputs
    for i in range(0, len(args.hidden_layers_layout)):
        kernel_initializer = build_initializer(args.weight_initializers[i]) if i < len(args.weight_initializers) else None
        bias_initializer = build_initializer(args.bias_initializers[i]) if i < len(args.bias_initializers) else None
        hidden = Dense(
            args.hidden_layers_layout[i],
            use_bias = True,
            activation = build_activation_function(args.activation_functions[i]),
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer
            )(hidden)
    outputs = []
    for i in range(0, n_output):
        output = Dense(1)(hidden)
        outputs.append(output)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_initializer(init):
    exp_init = 'lambda _ : tfi.' + init
    return eval(exp_init)(None)

def build_activation_function(af):
    if af.lower() == 'none':
        return None
    exp_af = 'lambda _ : tka.' + af
    return eval(exp_af)(None)

def build_optimizer():
    opt_init = args.optimizer
    exp_po = 'lambda _ : tko.' + opt_init
    optimizer = eval(exp_po)(None)
    return optimizer

def build_loss():
    exp_loss = 'lambda _ : tkl.' + args.loss
    return eval(exp_loss)(None)

def read_timeseries(tsfilename):
    y_values = []
    with open(tsfilename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            y_values.append(float(row[0]))
    return y_values

class EpochLogger(tfcb.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if  (epoch % args.model_snapshots_freq == 0) or ((epoch + 1) == args.epochs):
            self.model.save(os.path.join(args.model_snapshots_path, format(epoch, '09')))
            print ('\nSaved #{} snapshot model'.format(epoch, '09'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mpts_mlp_shmo_fit.py builds a model to fit multiple parallel time series using a single-headed multiple-output configurable MLP neural network')

    #todo add --version

    parser.add_argument('--traintslist',
                        type=str,
                        nargs = '+',
                        dest='train_timeseries_filenames',
                        required=True,
                        help='parallel time series file list (csv format) to train')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='output model directory')

    parser.add_argument('--samplelength',
                        type=int,
                        dest='sample_length',
                        required=False,
                        default=5,
                        help='sample length')

    #parser.add_argument('--valds',
    #                    type=str,
    #                    dest='val_dataset_filename',
    #                    required=False,
    #                    help='validation dataset file (csv format)')

    parser.add_argument('--bestmodelmonitor',
                        type=str,
                        dest='best_model_monitor',
                        required=False,
                        help='quantity to monitor in order to save the best model')

    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        required=False,
                        default=500,
                        help='number of epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        required=False,
                        default=50,
                        help='batch size')

    parser.add_argument('--hlayers',
                        type=int,
                        nargs = '+',
                        dest='hidden_layers_layout',
                        required=False,
                        default=[100],
                        help='number of neurons for each hidden layers')

    parser.add_argument('--hactivations',
                        type=str,
                        nargs = '+',
                        dest='activation_functions',
                        required=False,
                        default=['relu'],
                        help='activation functions between layers')

    parser.add_argument('--winitializers',
                        type=str,
                        nargs = '+',
                        dest='weight_initializers',
                        required=False,
                        default=[],
                        help='list of initializers (one for each layer) of the weights')

    parser.add_argument('--binitializers',
                        type=str,
                        nargs = '+',
                        dest='bias_initializers',
                        required=False,
                        default=[],
                        help='list of initializers (one for each layer) of the bias')

    parser.add_argument('--optimizer',
                        type=str,
                        dest='optimizer',
                        required=False,
                        default='Adam()',
                        help='optimizer algorithm')

    parser.add_argument('--loss',
                        type=str,
                        dest='loss',
                        required=False,
                        default='MeanSquaredError()',
                        help='loss function name')

    parser.add_argument('--metrics',
                        type=str,
                        nargs = '+',
                        dest='metrics',
                        required=False,
                        default=[],
                        help='list of metrics to compute')

    parser.add_argument('--dumpout',
                        type=str,
                        dest='dumpout_path',
                        required=False,
                        help='dump directory (directory to store loss and metric values)')

    parser.add_argument('--logsout',
                        type=str,
                        dest='logsout_path',
                        required=False,
                        help='logs directory for TensorBoard')

    parser.add_argument('--modelsnapout',
                        type=str,
                        dest='model_snapshots_path',
                        required=False,
                        help='output model snapshots directory')

    parser.add_argument('--modelsnapfreq',
                        type=int,
                        dest='model_snapshots_freq',
                        required=False,
                        default=25,
                        help='frequency in epochs to make the snapshot of model')

    args = parser.parse_args()

    if len(args.hidden_layers_layout) != len(args.activation_functions):
        raise Exception('Number of hidden layers and number of activation functions must be equals')

    print("#### Started mpts_mlp_shmo_fit ####");

    #todo: check for length
    sequences = []
    for train_timeseries_filename in args.train_timeseries_filenames:
        sequence = np.array(read_timeseries(train_timeseries_filename))
        sequence.shape = (len(sequence), 1)
        sequences.append(sequence)

    sequences = np.hstack(tuple(sequences))

    #validation_data = None
    #if args.val_dataset_filename is not None:
    #    validation_data = read_dataset(args.val_dataset_filename)

    #print(sequences.shape)
    #print(sequences)
    X_train, y_train = build_samples(sequences)
    #for i in range(len(X_train)):
    #    print(X_train[i], y_train[i])

    n_input = X_train.shape[1] * X_train.shape[2];
    n_output = sequences.shape[1];
    X_train = X_train.reshape((X_train.shape[0], n_input))

    y_train_old = y_train
    y_train = []
    for i in range(0, n_output):
        y_train.append(y_train_old[:, i].reshape((y_train_old.shape[0], 1)))

    model = build_model(n_input, n_output)

    optimizer = build_optimizer()
    loss=build_loss()
    model.compile(loss=loss, optimizer=optimizer, metrics = args.metrics)
    model.summary()

    tf_callbacks = []
    if args.logsout_path:
        tf_callbacks.append(tfcb.TensorBoard(log_dir=args.logsout_path, histogram_freq=0, write_graph=True, write_images=True))
    if args.model_snapshots_path:
        tf_callbacks.append(EpochLogger())
    if args.best_model_monitor:
        tf_callbacks.append(tfcb.ModelCheckpoint(
            filepath = args.model_path,
            save_best_only = True,
            monitor = args.best_model_monitor,
            mode = 'auto',
            verbose=1))

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        #validation_data=validation_data,
        callbacks=tf_callbacks)
    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    if not args.best_model_monitor:
        model.save(args.model_path)
        print ('\nSaved last recent model')

    if args.dumpout_path is not None:
        if not os.path.exists(args.dumpout_path):
            os.makedirs(args.dumpout_path)
        np.savetxt(os.path.join(args.dumpout_path, 'loss_' + loss.name + '.csv'), history.history['loss'], delimiter=',')
        for metric in args.metrics:
            np.savetxt(os.path.join(args.dumpout_path, 'metric_' + metric + '.csv'), history.history[metric], delimiter=',')
            #if validation_data is not None:
            #    np.savetxt(os.path.join(args.dumpout_path, 'val_' + metric + '.csv'), history.history['val_' + metric], delimiter=',')

    print("#### Terminated mpts_mlp_shmo_fit ####");
