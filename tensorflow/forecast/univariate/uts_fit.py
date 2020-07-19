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
import tensorflow.keras.layers as tfl
import tensorflow.keras.models as tfm
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_samples(seq):
    df = pd.DataFrame(seq)
    cols = list()
    for i in range(args.sample_length, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, 1):
        cols.append(df.shift(-i))

    aggregate = pd.concat(cols, axis=1)
    aggregate.dropna(inplace=True)

    X_train, y_train = aggregate.values[:, :-1], aggregate.values[:, -1]

    is_there_convlstm_layer = len(args.convlstm_layers_layout) > 0
    is_there_cnn_layer = len(args.cnn_layers_layout) > 0
    is_there_lstm_layer = len(args.lstm_layers_layout) > 0

    if is_there_convlstm_layer:
        X_train = X_train.reshape((X_train.shape[0], args.sub_sample_length, 1, args.sample_length // args.sub_sample_length, 1))
    elif is_there_cnn_layer and is_there_lstm_layer:
        X_train = X_train.reshape((X_train.shape[0], args.sub_sample_length, args.sample_length // args.sub_sample_length, 1))
    elif is_there_cnn_layer or is_there_lstm_layer:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    else:
        input_shape = (args.sample_length,)

    return X_train, y_train

def build_convlstm_layer(convlstm_layer_layout):
    if convlstm_layer_layout.startswith('convlstm'):
        tupla_par = '(' + convlstm_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if len(tupla_par) == 5:
            filters, kernel_size, activation, kinit, binit = tupla_par
            kinit = build_initializer(kinit)
            binit = build_initializer(binit)
        elif len(tupla_par) == 4:
            filters, kernel_size, activation, kinit = tupla_par
            kinit = build_initializer(kinit)
            binit = 'zeros'
        elif len(tupla_par) == 3:
            filters, kernel_size, activation = tupla_par
            kinit, binit ='glorot_uniform', 'zeros'
        else:
            raise Exception('Wrong convlstm syntax for \'%s\'' % convlstm_layer_layout)
        convlstm_layer = tfl.ConvLSTM2D(
            filters=filters,
            kernel_size=(1, kernel_size),
            activation=activation,
            kernel_initializer=kinit,
            bias_initializer=binit)
    elif convlstm_layer_layout.startswith('dropout'):
        tupla_par = '(' + convlstm_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if isinstance(tupla_par, float):
            rate = tupla_par
        else:
            raise Exception('Wrong convlstm syntax for \'%s\'' % convlstm_layer_layout)
        convlstm_layer = tfl.Dropout(rate=rate)
    else:
        raise Exception('Unsupported convlstm layer layout \'%s\'' % convlstm_layer_layout)

    return convlstm_layer

def build_cnn_layer(cnn_layer_layout, wrap_with_time_distributed):
    if cnn_layer_layout.startswith('conv'):
        tupla_par = '(' + cnn_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if len(tupla_par) == 5:
            filters, kernel_size, activation, kinit, binit = tupla_par
            kinit = build_initializer(kinit)
            binit = build_initializer(binit)
        elif len(tupla_par) == 4:
            filters, kernel_size, activation, kinit = tupla_par
            kinit = build_initializer(kinit)
            binit = 'zeros'
        elif len(tupla_par) == 3:
            filters, kernel_size, activation = tupla_par
            kinit, binit ='glorot_uniform', 'zeros'
        else:
            raise Exception('Wrong cnn syntax for \'%s\'' % cnn_layer_layout)
        cnn_layer = tfl.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=kinit,
            bias_initializer=binit)
    elif cnn_layer_layout.startswith('maxpool'):
        tupla_par = '(' + cnn_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if isinstance(tupla_par, int):
            pool_size = tupla_par
        else:
            raise Exception('Wrong cnn syntax for \'%s\'' % cnn_layer_layout)
        cnn_layer = tfl.MaxPooling1D(pool_size=pool_size)
    elif cnn_layer_layout.startswith('dropout'):
        tupla_par = '(' + cnn_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if isinstance(tupla_par, float):
            rate = tupla_par
        else:
            raise Exception('Wrong cnn syntax for \'%s\'' % cnn_layer_layout)
        cnn_layer = tfl.Dropout(rate=rate)
    else:
        raise Exception('Unsupported cnn layer layout \'%s\'' % cnn_layer_layout)

    if wrap_with_time_distributed:
        cnn_layer = tfl.TimeDistributed(cnn_layer)
    return cnn_layer

def build_lstm_layer(lstm_layer_layout):
    if lstm_layer_layout.startswith('lstm'):
        tupla_par = '(' + lstm_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if len(tupla_par) == 4:
            units, activation, kinit, binit = tupla_par
            kinit = build_initializer(kinit)
            binit = build_initializer(binit)
        elif len(tupla_par) == 3:
            units, activation, kinit = tupla_par
            kinit = build_initializer(kinit)
            binit = 'zeros'
        elif len(tupla_par) == 2:
            units, activation = tupla_par
            kinit, binit ='glorot_uniform', 'zeros'
        else:
            raise Exception('Wrong lstm syntax for \'%s\'' % lstm_layer_layout)
        lstm_layer = tfl.LSTM(
            units=units,
            activation=activation,
            kernel_initializer=kinit,
            bias_initializer=binit)
    elif lstm_layer_layout.startswith('dropout'):
        tupla_par = '(' + lstm_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if isinstance(tupla_par, float):
            rate = tupla_par
        else:
            raise Exception('Wrong lstm syntax for \'%s\'' % lstm_layer_layout)
        lstm_layer = tfl.Dropout(rate=rate)
    else:
        raise Exception('Unsupported lstm layer layout \'%s\'' % lstm_layer_layout)

    return lstm_layer

def build_dense_layer(dense_layer_layout):
    if dense_layer_layout.startswith('dense'):
        tupla_par = '(' + dense_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if len(tupla_par) == 4:
            units, activation, kinit, binit = tupla_par
            kinit = build_initializer(kinit)
            binit = build_initializer(binit)
        elif len(tupla_par) == 3:
            units, activation, kinit = tupla_par
            kinit = build_initializer(kinit)
            binit = 'zeros'
        elif len(tupla_par) == 2:
            units, activation = tupla_par
            kinit, binit ='glorot_uniform', 'zeros'
        else:
            raise Exception('Wrong dense syntax for \'%s\'' % dense_layer_layout)
        dense_layer = tfl.Dense(
            units=units,
            activation=activation,
            kernel_initializer=kinit,
            bias_initializer=binit)
    elif dense_layer_layout.startswith('dropout'):
        tupla_par = '(' + dense_layer_layout.split('(', 1)[1]
        tupla_par = eval(tupla_par)
        if isinstance(tupla_par, float):
            rate = tupla_par
        else:
            raise Exception('Wrong dense syntax for \'%s\'' % dense_layer_layout)
        dense_layer = tfl.Dropout(rate=rate)
    else:
        raise Exception('Unsupported dense layer layout \'%s\'' % dense_layer_layout)

    return dense_layer

def build_model():
    is_there_convlstm_layer = len(args.convlstm_layers_layout) > 0
    is_there_cnn_layer = len(args.cnn_layers_layout) > 0
    is_there_lstm_layer = len(args.lstm_layers_layout) > 0

    if is_there_convlstm_layer:
        input_shape = (args.sub_sample_length, 1, args.sample_length // args.sub_sample_length, 1)
    elif is_there_cnn_layer and is_there_lstm_layer:
        input_shape = (None, args.sample_length // args.sub_sample_length, 1)
    elif is_there_cnn_layer or is_there_lstm_layer:
        input_shape = (args.sample_length, 1)
    else:
        input_shape = (args.sample_length,)

    inputs = tfl.Input(shape=input_shape)
    hidden = inputs

    for i in range(0, len(args.convlstm_layers_layout)):
        hidden = build_convlstm_layer(args.convlstm_layers_layout[i])(hidden)

    for i in range(0, len(args.cnn_layers_layout)):
        hidden = build_cnn_layer(args.cnn_layers_layout[i], is_there_lstm_layer)(hidden)

    if is_there_convlstm_layer:
        hidden = tfl.Flatten()(hidden)
    elif is_there_cnn_layer:
        if is_there_lstm_layer:
            hidden = tfl.TimeDistributed(tfl.Flatten())(hidden)
        else:
            hidden = tfl.Flatten()(hidden)

    for i in range(0, len(args.lstm_layers_layout)):
        hidden = build_lstm_layer(args.lstm_layers_layout[i])(hidden)

    for i in range(0, len(args.dense_layers_layout)):
        hidden = build_dense_layer(args.dense_layers_layout[i])(hidden)

    outputs = tfl.Dense(1)(hidden)
    model = tfm.Model(inputs=inputs, outputs=outputs)
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
    parser = argparse.ArgumentParser(description='%(prog)s builds a model to fit an univariate time series using a configurable LSTM neural network')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--tstrain',
                        type=str,
                        dest='train_timeseries_filename',
                        required=True,
                        help='univariate time series file (csv format) for training')

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

    parser.add_argument('--subsamplelength',
                        type=int,
                        dest='sub_sample_length',
                        required=False,
                        default=1,
                        help='sub sample length (used when both cnn and lstm layers are present in the model, otherwise ignored)')

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

    parser.add_argument('--batchsize',
                        type=int,
                        dest='batch_size',
                        required=False,
                        default=50,
                        help='batch size')

    parser.add_argument('--cnnlayers',
                        type=str,
                        nargs = '+',
                        dest='cnn_layers_layout',
                        required=False,
                        default=[],
                        help='CNN layer layout')

    parser.add_argument('--lstmlayers',
                        type=str,
                        nargs = '+',
                        dest='lstm_layers_layout',
                        required=False,
                        default=[],
                        help='LSTM layer layout')

    parser.add_argument('--convlstmlayers',
                        type=str,
                        nargs = '+',
                        dest='convlstm_layers_layout',
                        required=False,
                        default=[],
                        help='ConvLSTM layer layout')


    parser.add_argument('--denselayers',
                        type=str,
                        nargs = '+',
                        dest='dense_layers_layout',
                        required=False,
                        default=[],
                        help='dense layer layout')

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

    print("#### Started %s ####" % os.path.basename(__file__));

    sequence = read_timeseries(args.train_timeseries_filename)
    X_train, y_train = build_samples(sequence)

    model = build_model()

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

    print("#### Terminated %s ####" % os.path.basename(__file__));
