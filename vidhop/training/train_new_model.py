from logging import warning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.utils import class_weight as clw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sklearn.metrics as metrics
import scipy.stats
import time
import matplotlib
import tensorflow as tf
# from vidhop.cli import LazyLoader
# tf = LazyLoader('tensorflow')

matplotlib.rcParams['backend'] = 'Agg'
import matplotlib.pyplot as plt
from vidhop.DataParsing import DataParsing_main
import pickle
import click


@click.command(short_help="train a model on your training files generated with make_dataset")
@click.option('--inpath', '-i', required=True,
              help='path to the dir with training files, generated with make_dataset.py')
@click.option('--outpath', '-o', default='.', help='path where to save the output')
@click.option('--name', '-n', help='suffix added to output file names')
@click.option('--epochs', '-e', default=int(100), help='number of epochs for training the model')
@click.option('--architecture', '-a', default=int(1), help='select architecture (0:LSTM, 1:CNN+LSTM)')
@click.option('--extention_variant', '-v', default=int(1),
              help='select extension variant (0:Normal repeat, 1:Normal repeat with gaps, 2:Random repeat, 3:Random repeat with gaps, 4:Append gaps, 5:Smallest, 6:Online)')
@click.option('--early_stopping', '-s', is_flag=True,
              help='stop training when model accuracy did not improve over time, patience 5% of max epochs')
@click.option('--repeated_undersampling', '-r', is_flag=True,
              help='use repeated undersampling while training, to be usable the training files must be generated with asdfasres.py and activated reapeted undersampling parameter')
def training(inpath, outpath, name, epochs, architecture, extention_variant, early_stopping, repeated_undersampling):
    ''' Train a model on your training files generated with make_dataset

        \b
        Example:
        set input output and name of the model
        $ python train_new_model.py -i /home/user/trainingdata/ -o /home/user/model/ --name test
        \b
        use the LSTM archtecture and the extention variant random repeat
        $ python train_new_model.py -i /home/user/trainingdata/ --architecture 0 --extention_variant 2
        \b
        use repeated undersampling for training, note that for this the dataset must have been created with repeated undersampling enabled
        $ python train_new_model.py -i /home/user/trainingdata/ -r
        \b
        train the model for 40 epochs, stop training if for 2 epochs the accuracy did not increase
        $ python train_new_model.py -i /home/user/trainingdata/ --epochs 40 --early_stopping
        '''
    if extention_variant in (0, 1, 2, 3):
        repeat = True
    else:
        repeat = False

    if extention_variant in (2, 3):
        randomrepeat = True
    else:
        randomrepeat = False

    if extention_variant in (1, 3):
        use_repeat_spacer = True
    else:
        use_repeat_spacer = False

    if extention_variant == 5:
        kwargs = dict({"maxLen": -1, "input_subSeqlength": 0})
    else:
        kwargs = dict()

    if extention_variant == 6:
        online_training = True
    else:
        online_training = False

    if architecture == 0:
        design = 4
    else:
        design = 7

    files = os.listdir(inpath)
    assert "Y_train.csv" in files, f"{inpath} must contain Y_train.csv file, but no such file in {files}"

    test_and_plot(inpath=inpath, outpath=outpath, suffix=name, online_training=online_training, repeat=repeat,
                  randomrepeat=randomrepeat, early_stopping_bool=early_stopping, do_shrink_timesteps=True,
                  use_repeat_spacer=use_repeat_spacer, design=design, nodes=150, faster=True,
                  use_generator=repeated_undersampling, epochs=epochs, dropout=0.2, accuracy=True, **kwargs)



class lrManipulator(tf.keras.callbacks.Callback):
    """
    Manipulate the lr for Adam Optimizer
    -> no big chances usefull
    """

    def __init__(self, nb_epochs, nb_snapshots):
        self.T = nb_epochs
        self.M = nb_snapshots

    def on_epoch_begin(self, epoch, logs={}):
        tf.keras.backend.set_value(self.model.optimizer.lr, 0.001)
        if ((epoch % (self.T // self.M)) == 0):
            tf.keras.backend.set_value(self.model.optimizer.iterations, 0)
            tf.keras.backend.set_value(self.model.optimizer.lr, 0.01)


class TimeHistory(tf.keras.callbacks.Callback):
    """https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit"""

    def on_train_begin(self, logs={}):
        if not hasattr(self, 'times'):
            self.times = []
            self.time_train_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        logs = logs or {}
        self.times.append(int(time.time()) - int(self.time_train_start))


prediction_val = []


class accuracyHistory(tf.keras.callbacks.Callback):
    """to get the accuracy of my personal voting scores"""

    def on_train_begin(self, logs={}):
        if not hasattr(self, 'meanVote_val'):
            self.meanVote_val = []
            self.normalVote_val = []

    def on_epoch_begin(self, epoch, logs=None):
        global prediction_val
        prediction_val = []

    def on_epoch_end(self, batch, logs={}):
        """
        1. make prediction of train
        2. get the voting results
        3. calc and save accuracy
        4. do same for val set
        """
        logs = logs or {}
        global prediction_val

        if (len(prediction_val) == 0):
            prediction_val = (self.model.predict(X_val))

        self.prediction_val = prediction_val

        y_true_small, y_pred_mean_val, y_pred_voted_val, y_pred, y_pred_mean_exact = \
            calc_predictions(X_val, Y_val, do_print=False, y_pred=self.prediction_val)
        self.normalVote_val.append(metrics.accuracy_score(y_true_small, y_pred_voted_val))
        self.meanVote_val.append(metrics.accuracy_score(y_true_small, y_pred_mean_val))


class roc_History(tf.keras.callbacks.Callback):
    """to get the AUC of my personal voting scores"""

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def on_train_begin(self, logs={}):
        if not hasattr(self, 'roc_val'):
            # roc curve values for validation set
            self.roc_macro = []
            # roc curve values of the joined subsequences for the validation set
            self.roc_mean_val = []
            # roc curve values of the vote of the subsequences for the validation set
            self.roc_meanVote_val = []
            # thresholds per class
            self.thresholds = []
            # accuracy with general threshold tuning
            self.acc_val_threshold_tuned = []
            # accuracy with multi-threshold tuning
            self.acc_val_multi_thresholds_tuned = []

    def on_epoch_begin(self, epoch, logs=None):
        global prediction_val
        prediction_val = []

    def on_epoch_end(self, batch, logs={}):
        """
        1. make prediction of train
        2. get the voting results
        3. calc and save accuracy
        4. do same for val set
        """
        logs = logs or {}

        # check if allready calculated validation results, if no calc new
        global prediction_val
        if (len(prediction_val) == 0):
            prediction_val = (self.model.predict(X_val))
        self.prediction_val = prediction_val

        y_true_small, y_pred_mean_val, y_pred_voted_val, y_pred, y_pred_mean_val_exact = \
            calc_predictions(X_val, Y_val, do_print=False, y_pred=self.prediction_val)
        n_classes = Y_val.shape[-1]
        y_true_small_bin = tf.keras.utils.to_categorical(y_true_small, n_classes)
        y_pred_mean_val_bin = tf.keras.utils.to_categorical(y_pred_mean_val, n_classes)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        acc = dict()
        cutoffs = np.array([])
        thresh_tpr_values = np.array([])
        thresh_fpr_values = np.array([])

        for i in range(n_classes):
            fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_true_small_bin[:, i], y_pred_mean_val_exact[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            acc_i = []
            thresholds_interp = np.linspace(0, 1, 101)
            for cutoff in thresholds_interp:
                acc_i.append(acc_with_cutoffs_per_class(y_true_small_bin[:, i], y_pred_mean_val_exact[:, i], cutoff))
            acc[i] = acc_i
            """alternative variants
            # calc optimal accuracy threshold per class
            # optimal_idx = np.argmax(np.abs(tpr[i] - fpr[i]))
            # optimal = tpr[i] / (tpr[i] + fpr[i])
            # fnr = 1-tpr[i]
            # tnr = 1-fpr[i]
            # precision = tpr[i] / (tpr[i] + fpr[i])
            # recall = tpr[i] / 1
            # optimal = 2*((precision*recall)/(precision+recall))
            # acc = (tpr[i] + (1 - fpr[i])) / 2
            # optimal = tpr[i]/(tpr[i]+fpr[i]+fnr)
            # optimal[np.isnan(optimal)] = 0
            # optimal_idx = np.argmax(optimal)
            # optimal_threshold = thresholds[i][optimal_idx]
            """
            # use highest threshold, so need to reverse acc
            optimal_idx = np.argmax(acc[i][::-1])
            optimal_idx = len(acc[i]) - optimal_idx - 1
            optimal_threshold = thresholds_interp[optimal_idx]
            # muss thresholds umdrehen da in interp() x aufsteigend sein muss
            thresh_fpr = np.interp(optimal_threshold, thresholds[i][::-1], fpr[i][::-1])
            thresh_tpr = np.interp(optimal_threshold, thresholds[i][::-1], tpr[i][::-1])
            cutoffs = np.append(cutoffs, optimal_threshold)
            thresh_tpr_values = np.append(thresh_tpr_values, thresh_tpr)
            thresh_fpr_values = np.append(thresh_fpr_values, thresh_fpr)

        # print(f"cutoffs: {cutoffs}")
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true_small_bin.ravel(), y_pred_mean_val_exact.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        mean_thresholds = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_thresholds += np.interp(all_fpr, fpr[i], thresholds[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        mean_thresholds /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        optimal_idx = np.argmax(np.abs(mean_tpr - all_fpr))
        general_optimal_threshold = mean_thresholds[optimal_idx]

        plt.scatter(all_fpr[optimal_idx], mean_tpr[optimal_idx], c="green",
                    label=f"best general threshold: {round(general_optimal_threshold, 4)}")
        plt.scatter(thresh_fpr_values.mean(), thresh_tpr_values.mean(), c="red",
                    label=f"multi threshold tuned")

        self.roc_macro.append(roc_auc["macro"])
        self.roc_mean_val.append(metrics.roc_auc_score(y_true_small_bin, y_pred_mean_val_exact))
        self.roc_meanVote_val.append(metrics.roc_auc_score(y_true_small_bin, y_pred_mean_val_bin))
        self.thresholds.append(cutoffs)

        self.acc_val_threshold_tuned.append(acc_with_cutoffs(Y_true=y_true_small_bin, Y_pred=y_pred_mean_val_exact,
                                                             cutoffs=[general_optimal_threshold] * n_classes))
        self.acc_val_multi_thresholds_tuned.append(
            acc_with_cutoffs(Y_true=y_true_small_bin, Y_pred=y_pred_mean_val_exact, cutoffs=cutoffs))
        print(f"roc_macro: {self.roc_macro[-1]}")
        print(f"roc_mean_val: {self.roc_mean_val[-1]}")
        print(f"roc_meanVote_val: {self.roc_meanVote_val[-1]}")
        print(f"thresholds: {self.thresholds[-1]}")
        print(f"acc_val_threshold_tuned: {self.acc_val_threshold_tuned[-1]}")
        print(f"acc_val_multi_thresholds_tuned: {self.acc_val_multi_thresholds_tuned[-1]}")

        fpr, tpr, _ = metrics.roc_curve(y_true_small_bin.ravel(), y_pred_mean_val_bin.ravel())
        plt.plot(fpr, tpr, label='mean vote ROC curve (area = {0:0.2f})'
                                 ''.format(self.roc_meanVote_val[-1]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multiple classes')
        plt.legend(loc="lower right")
        plt.savefig(self.path + f"/roc_curve_{self.name}.pdf")
        print(f"save to {self.path}/roc_curve_{self.name}.pdf")
        plt.clf()
        plt.close()

        print(
            f"roc macro: {self.roc_macro[-1]}\n roc mean val: {self.roc_mean_val[-1]}\n roc mean Vote val: {self.roc_meanVote_val[-1]}")


class prediction_history(tf.keras.callbacks.Callback):
    """Callback subclass that prints each epoch prediction"""

    def on_epoch_end(self, epoch, logs={}):
        p = np.random.permutation(len(Y_val))
        shuffled_X = X_val[p]
        shuffled_Y = Y_val[p]
        self.predhis = (self.model.predict(shuffled_X[0:10]))
        y_pred = np.argmax(self.predhis, axis=-1)
        y_true = np.argmax(shuffled_Y, axis=-1)[0:10]
        print(f"Predicted: {y_pred}")
        print(f"True:      {y_true}")
        table = pd.crosstab(
            pd.Series(y_true),
            pd.Series(y_pred),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print(table)


class History(tf.keras.callbacks.Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
            self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class StopEarly(tf.keras.callbacks.Callback):
    """
    Callback that stops training after an epoch
    important for online training
    """

    def on_epoch_end(self, epoch, logs=None):
        self.model.stop_training = True


def acc_with_cutoffs(Y_true, Y_pred, cutoffs):
    assert Y_true.shape[1] == len(cutoffs), "number of cutoffs and classes must be equal"
    # binarize the pred with cutoffs
    Y_pred_bin = np.zeros(Y_true.shape)
    for i in range(0, Y_true.shape[1]):
        Y_pred_bin[:, i] = Y_pred[:, i] > cutoffs[i]

    for index, sample in enumerate(Y_pred_bin):
        # if no entry over threshold use highest value as prediction
        if sum(sample) == 0:
            Y_pred_bin[index, np.argmax(sample)] = 1

    # find multiple predictions
    positions = np.argwhere(Y_pred_bin == 1)
    Y_true_extended = np.array([])
    Y_pred_extended = np.array([])
    for i in positions:
        true = np.argmax(Y_true[i[0]])
        pred = i[1]
        Y_true_extended = np.append(Y_true_extended, true)
        Y_pred_extended = np.append(Y_pred_extended, pred)

    acc = sum(Y_true_extended == Y_pred_extended) / len(Y_true_extended)
    print(f"threshold tuned accuracy: {acc}")
    return acc


def acc_with_cutoffs_per_class(Y_true, Y_pred, cutoffs):
    # binarize the pred with cutoffs
    # Y_pred_bin = np.zeros(Y_true.shape)
    Y_pred_bin = Y_pred > cutoffs
    prediction_mask = np.array(Y_true + Y_pred_bin, dtype=np.bool)

    acc = sum(Y_true[prediction_mask] == Y_pred_bin[prediction_mask]) / Y_true[
        prediction_mask].size  # same as below but not as intuitive
    # acc2 = sum(Y_true == Y_pred_bin)/Y_true.size #this biases towards negative prediction
    return acc


def use_old_data(one_hot_encoding=True):
    """
    to reuse the "old" exported data
    """

    Y_train_old = np.genfromtxt(directory + '/Y_train.csv', delimiter=',', dtype='str')
    Y_test_old = np.genfromtxt(directory + '/Y_test.csv', delimiter=',', dtype='str')
    X_train_old = np.genfromtxt(directory + '/X_train.csv', delimiter=',', dtype='int16')
    X_test_old = np.genfromtxt(directory + '/X_test.csv', delimiter=',', dtype='int16')

    def one_hot_encode_int(data):

        """
        One hot encoding
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        """

        num_classes = np.max(data) + 1
        encoded_data = tf.keras.utils.to_categorical(data, num_classes=num_classes)
        encoded_data = encoded_data.reshape((data.shape[0], data.shape[1], num_classes))
        return encoded_data

    def one_hot_encode_string(y):

        """
        One hot encoding
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        """
        encoder = LabelEncoder()
        encoder.fit(y)
        print(encoder.classes_)
        print(encoder.transform(encoder.classes_))
        encoded_Y = encoder.transform(y)
        return tf.keras.utils.to_categorical(encoded_Y)

    if one_hot_encoding:
        global X_test, X_train, Y_test, Y_train
        X_test = one_hot_encode_int(X_test_old)
        X_train = one_hot_encode_int(X_train_old)
    else:
        X_test = X_test_old
        X_train = X_train_old

    Y_test = one_hot_encode_string(Y_test_old)
    Y_train = one_hot_encode_string(Y_train_old)


def use_data_nanocomb(directory, one_hot_encoding=True, repeat=True, use_spacer=True, maxLen=0, online=False,
                      unbalanced=False):
    """
    to use the nanocomb exported data
    """

    Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_train_old = pd.read_csv(directory + '/X_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].values

    create_val = False

    try:
        Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None)[1].values
        X_val_old = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None)[1].values
        print("loaded validation set from: " + directory + '/Y_val.csv')
    except:
        print("create validation set from train")
        create_val = True

    if create_val:
        assert unbalanced == False, "an unbalanced training set needs a predefined validation set"
        X_train_old, X_val_old, Y_train_old, Y_val_old = train_test_split(X_train_old, Y_train_old, test_size=0.3,
                                                                          random_state=SEED,
                                                                          stratify=Y_train_old)
        create_val = False

    if one_hot_encoding:
        global X_test, X_train, X_val, Y_test, Y_train, Y_val
        if maxLen <= 0:
            length = []
            x_sets = [X_test_old, X_train_old]
            if create_val == False:
                x_sets.append(X_val_old)

            for X in x_sets:
                for i in X:
                    length.append(len(i))
            length.sort()
            # plt.hist(length,bins=500,range=(0,20000))
            # plt.show()
            print(f"shortest sequence = {length[0]}")
            if maxLen == -1:
                maxLen = length[0]
            else:
                maxLen = length[int(len(length) * 0.95)]
            print(f"maxLen = {maxLen}")

        # def encode_string(maxLen=None, x=[], y=[], y_encoder=None, repeat=True, use_spacer=True, online_Xtrain_set=False):
        X_train = DataParsing_main.encode_string(maxLen=maxLen, x=X_train_old, repeat=repeat, use_spacer=use_spacer,
                                                 online_Xtrain_set=online)
        X_test = DataParsing_main.encode_string(maxLen=maxLen, x=X_test_old, repeat=repeat, use_spacer=use_spacer)
        if create_val == False:
            X_val = DataParsing_main.encode_string(maxLen=maxLen, x=X_val_old, repeat=repeat, use_spacer=use_spacer)
    else:
        X_train = X_train_old
        X_test = X_test_old
        if create_val == False:
            X_val = X_val_old

    Y_train, y_encoder = DataParsing_main.encode_string(y=Y_train_old)
    Y_test = DataParsing_main.encode_string(y=Y_test_old, y_encoder=y_encoder)
    Y_val = DataParsing_main.encode_string(y=Y_val_old, y_encoder=y_encoder)
    return maxLen


def filter_train_data(species_to_keep=[1, 2]):
    """
    to define which classes should be learned
    :param species_to_keep: array with classes/labels which should be included in the train data e.g. [0,2]
    :return:
    """
    global X_train, Y_train
    Y_train_int = np.argmax(Y_train, axis=-1)
    arr = np.zeros(X_train.shape[0], dtype=int)
    for species in species_to_keep:
        arr[Y_train_int == int(species)] = 1
    X_train = X_train[arr == 1, :]
    Y_train = Y_train[arr == 1]


def model_for_plot(inpath, outpath, design=1, sampleSize=1, nodes=32, suffix="", epochs=100, dropout=0,
                   faster=False, voting=False, tensorboard=False, early_stopping_bool=True,
                   shuffleTraining=True, batch_norm=False, online_training=False,
                   number_subsequences=1, use_generator=True, repeat=True, use_spacer=False, randomrepeat=False,
                   **kwargs):
    """
    method to train a model with specified properties, saves training behavior in /$path/"history"+suffix+".csv"
    :param design: parameter for complexity of the NN, 0 == 2 layer GRU, 1 == 2 layer LSTM, 2 == 3 layer LSTM
    :param sampleSize: fraction of samples that will be used for training (1/samplesize). 1 == all samples, 2 == half of the samples
    :param nodes: number of nodes per layer
    :param suffix: suffix for output files
    :param epochs: number of epochs to train
    :param dropout: rate of dropout to use, 0 == no Dropout, 0.2 = 20% Dropout
    :param timesteps: size of "memory" of LSTM, don't change if not sure what you're doing
    :param faster: speedup due higher batch size, can reduce accuracy
    :param outpath: define the directory where the training history should be saved
    :param voting: if true than saves the history of the voting / mean-predict subsequences, reduces training speed
    :param tensorboard: for observing live changes to the network, more details see web
    :param cuda: use GPU for calc, not tested jet, not working
    :return: dict with loss and model
    """
    model = tf.keras.models.Sequential()
    global batch_size, X_train, X_test, Y_train

    # Y_train_noOHE = np.argmax(Y_train, axis=1)
    if use_generator:
        class_weight = None

    else:
        Y_train_noOHE = [y.argmax() for y in Y_train]
        class_weight = clw.compute_class_weight('balanced', np.unique(Y_train_noOHE), Y_train_noOHE)
        class_weight_dict = {i: class_weight[i] for i in range(len(class_weight))}
        class_weight = class_weight_dict
        print(f"class_weights: {class_weight}")

    timesteps = X_test.shape[1]

    if faster:
        batch = batch_size * 16
    else:
        batch = batch_size

    if design == 0:
        model.add(tf.keras.layers.GRU(nodes, input_shape=(timesteps, X_test.shape[-1]), return_sequences=True,
                                      dropout=dropout))
        model.add(tf.keras.layers.GRU(nodes, dropout=dropout))

    if design == 1:
        model.add(tf.keras.layers.LSTM(nodes, input_shape=(timesteps, X_test.shape[-1]), return_sequences=True,
                                       dropout=dropout))
        model.add(tf.keras.layers.LSTM(nodes, dropout=dropout))

    if design == 2:
        model.add(tf.keras.layers.LSTM(nodes, input_shape=(timesteps, X_test.shape[-1]), return_sequences=True,
                                       dropout=dropout))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(nodes, dropout=dropout))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())

    if design == 3:
        model.add(tf.keras.layers.LSTM(nodes, input_shape=(timesteps, X_test.shape[-1]), return_sequences=True,
                                       dropout=dropout))
        model.add(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout))
        model.add(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout))
        model.add(tf.keras.layers.LSTM(nodes, dropout=dropout))

    if design == 4:
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout),
                                                input_shape=(timesteps, X_test.shape[-1])))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, dropout=dropout)))

    if design == 5:
        model.add(tf.keras.layers.Conv1D(nodes, 9, input_shape=(timesteps, X_test.shape[-1]), activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Conv1D(nodes, 9, activation='relu'))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, dropout=dropout)))

    if design == 6:
        # This returns a tensor
        inputs = tf.keras.layers.Input(shape=(timesteps, X_test.shape[-1]))

        left1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout))(
            inputs)
        left2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, dropout=dropout))(left1)

        right = tf.keras.layers.Conv1D(nodes, 9, activation='relu')(inputs)
        right = tf.keras.layers.MaxPooling1D(3)(right)
        right = tf.keras.layers.Conv1D(nodes, 9, activation='relu')(right)
        right = tf.keras.layers.MaxPooling1D(3)(right)
        right3 = tf.keras.layers.Conv1D(nodes, 9, activation='relu')(right)
        right_flat = tf.keras.layers.Flatten()(right3)

        joined = tf.keras.layers.Concatenate()([left2, right_flat])
        predictions = tf.keras.layers.Dense(Y_train.shape[-1], activation='softmax')(joined)

        model = tf.keras.Model(inputs=inputs, outputs=predictions)

    if design == 7:
        model.add(tf.keras.layers.Conv1D(nodes, 9, input_shape=(timesteps, X_test.shape[-1])))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Conv1D(nodes, 9))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(nodes, return_sequences=True, recurrent_activation="sigmoid"),
            input_shape=(timesteps, X_test.shape[-1])))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, recurrent_activation="sigmoid")))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    if design == 8:
        model.add(tf.keras.layers.Conv1D(nodes, 9, input_shape=(timesteps, X_test.shape[-1]), activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Conv1D(nodes, 9, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Conv1D(nodes, 9, activation='relu'))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2)))

    if design != 6:
        model.add(tf.keras.layers.Dense(nodes, activation='elu'))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(Y_train.shape[-1], activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'], sample_weight_mode=None)
    # return model
    filepath = inpath + "/model_best_acc_" + suffix + ".hdf5"
    filepath2 = inpath + "/model_best_loss_" + suffix + ".hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                    mode='max')
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
                                                     mode='min')
    predictions = prediction_history()
    time_callback = TimeHistory()

    if early_stopping_bool:
        early_stopping = tf.keras.callbacks.EarlyStopping('val_acc', min_delta=0, patience=epochs // 20,
                                                          restore_best_weights=True, verbose=2)
        # early_stopping2 = EarlyStopping('val_loss', min_delta=0, patience=epochs//20,restore_best_weights=True)

        callbacks_list = [checkpoint, checkpoint2, predictions, time_callback, early_stopping]
    else:
        callbacks_list = [checkpoint, checkpoint2, predictions, time_callback]
    # callbacks_list = [early_stopping2, early_stopping, predictions, time_callback]

    if voting:
        myAccuracy = accuracyHistory()
        myRoc = roc_History(name=suffix, path=inpath)
        callbacks_list.append(myAccuracy)
        callbacks_list.append(myRoc)

    if tensorboard:
        if not os.path.isdir(outpath + '/my_log_dir'):
            os.makedirs(outpath + '/my_log_dir')
        tensorboard = tf.keras.callbacks.TensorBoard(
            # Log files will be written at this location
            log_dir=outpath + '/my_log_dir',
            # We will record activation histograms every 1 epoch
            histogram_freq=1,
            # We will record embedding data every 1 epoch
            embeddings_freq=1,
        )
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=outpath + '/my_log_dir', histogram_freq=0, batch_size=32,
                                                     write_graph=True, write_grads=False, write_images=False,
                                                     embeddings_freq=0, embeddings_layer_names=None,
                                                     embeddings_metadata=None)
        callbacks_list.append(tensorboard)

    if use_generator:
        from vidhop.training.DataGenerator import DataGenerator
        params = {"number_subsequences": number_subsequences, "dim": timesteps, "n_channels": X_test.shape[-1],
                  "number_samples_per_batch": batch_size,
                  "n_classes": Y_test.shape[-1], "shuffle": shuffleTraining, "online_training": online_training,
                  "seed": 1, "repeat": repeat, "use_spacer": use_spacer, "randomrepeat": randomrepeat, "faster": faster}

        # global directory
        training_generator = DataGenerator(directory=inpath + "/train", **params, **kwargs)

        hist = model.fit(training_generator, epochs=epochs, callbacks=callbacks_list, validation_data=(X_val, Y_val),
                         class_weight=class_weight, shuffle=shuffleTraining)
    else:
        if online_training == True:
            print("use online training")

            assert X_test.shape[2] == 6, "different base coding, not -,A,C,G,N,T"
            # print("expect -,A,C,G,N,T to categorial")

            # filter to small samples
            subSeqLength = timesteps
            index_clean = [i for i, j in enumerate(X_train) if len(j) > subSeqLength]
            X_train_copy_clean = [X_train[i] for i in index_clean]
            Y_train_copy_clean = [Y_train[i] for i in index_clean]

            Y_train = np.array(Y_train_copy_clean)

            hist = History()
            stop = StopEarly()
            callbacks_list.append(hist)
            callbacks_list.append(stop)

            for epo in range(epochs):
                X_train_manipulated, Y_train_manipulated = DataParsing_main.manipulate_training_data(
                    X=X_train_copy_clean,
                    Y=Y_train,
                    subSeqLength=subSeqLength,
                    number_subsequences=number_subsequences)
                # number_subsequences=batch_size)

                Y_train_noOHE = [y.argmax() for y in Y_train_manipulated[0:int(len(X_train_manipulated) / sampleSize)]]

                class_weight = clw.compute_class_weight('balanced', np.unique(Y_train_noOHE), Y_train_noOHE)
                class_weight_dict = {i: class_weight[i] for i in range(len(class_weight))}
                class_weight = class_weight_dict
                print(class_weight)

                model.fit(X_train_manipulated[0:int(len(X_train_manipulated) / sampleSize)],
                          Y_train_manipulated[0:int(len(X_train_manipulated) / sampleSize)],
                          epochs=epo + 1, batch_size=batch, callbacks=callbacks_list, initial_epoch=epo,
                          validation_data=(X_val, Y_val), class_weight=class_weight, shuffle=shuffleTraining)

        else:
            hist = model.fit(X_train[0:int(len(X_train) / sampleSize)], Y_train[0:int(len(X_train) / sampleSize)],
                             epochs=epochs, callbacks=callbacks_list, batch_size=batch,
                             validation_data=(X_val, Y_val), class_weight=class_weight,
                             # sample_weight=sample_weights,
                             shuffle=shuffleTraining)

    times = time_callback.times
    if voting:
        val_acc_votes = myAccuracy.normalVote_val
        val_acc_means = myAccuracy.meanVote_val
        val_roc_macro = myRoc.roc_macro
        val_roc_micro = myRoc.roc_mean_val
        val_roc_vote = myRoc.roc_meanVote_val
        val_acc_threshold = myRoc.acc_val_threshold_tuned
        val_acc_multi_threshold = myRoc.acc_val_multi_thresholds_tuned

    if not os.path.isfile(outpath + "/history" + suffix + ".csv"):
        histDataframe = pd.DataFrame(hist.history)
        cols = ["acc", "loss", "val_acc", "val_loss"]
        histDataframe = histDataframe[cols]
        histDataframe = histDataframe.assign(time=times)
        if voting:
            histDataframe = histDataframe.assign(val_acc_vote=val_acc_votes)
            histDataframe = histDataframe.assign(val_acc_mean=val_acc_means)
            histDataframe = histDataframe.assign(val_roc_micro=val_roc_micro)
            histDataframe = histDataframe.assign(val_roc_macro=val_roc_macro)
            histDataframe = histDataframe.assign(val_roc_vote=val_roc_vote)
            histDataframe = histDataframe.assign(val_acc_threshold=val_acc_threshold)
            histDataframe = histDataframe.assign(val_acc_multi_threshold=val_acc_multi_threshold)
        histDataframe.to_csv(outpath + "/history" + suffix + ".csv")
    else:
        histDataframe = pd.DataFrame(hist.history)
        histDataframe = histDataframe.assign(time=times)
        if voting:
            histDataframe = histDataframe.assign(val_acc_vote=val_acc_votes)
            histDataframe = histDataframe.assign(val_acc_mean=val_acc_means)
            histDataframe = histDataframe.assign(val_roc_micro=val_roc_micro)
            histDataframe = histDataframe.assign(val_roc_macro=val_roc_macro)
            histDataframe = histDataframe.assign(val_roc_vote=val_roc_vote)
            histDataframe = histDataframe.assign(val_acc_threshold=val_acc_threshold)
            histDataframe = histDataframe.assign(val_acc_multi_threshold=val_acc_multi_threshold)
        histDataframe.to_csv(outpath + "/history" + suffix + ".csv", mode='a', header=False)
    return


def calc_predictions(X, Y, y_pred, do_print=False):
    """
    plot predictions
    :param X: raw-data which should be predicted
    :param Y: true labels for X
    :param do_print: True == print the cross-tab of the prediction
    :param y_pred: array with predicted labels for X
    :return: y_true_small == True labels for complete sequences, yTrue == True labels for complete subsequences, y_pred_mean == with mean predicted labels for complete sequences, y_pred_voted == voted labels for complete sequences, y_pred == predicted labels for complete subsequences
    """

    def print_predictions(y_true, y_pred, y_true_small, y_pred_voted, y_pred_sum, y_pred_mean_weight_std,
                          y_pred_mean_weight_ent):

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true)),
            pd.Series(y_encoder.inverse_transform(y_pred)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("standard version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true, y_pred) * 100
        print("standard version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_voted)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("vote version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_voted) * 100
        print("vote version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_sum)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("mean version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_sum) * 100
        print("mean version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_mean_weight_ent)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("entropie version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_mean_weight_ent) * 100
        print("entropie version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_sum)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("std version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_mean_weight_std) * 100
        print("std-div version")
        print("acc = " + str(accuracy))

    # for addition of probability and not voting
    y_pred_mean = []
    y_pred_mean_exact = []
    weigth_entropy = []
    y_pred_mean_weight_ent = []
    weigth_std = []
    y_pred_mean_weight_std = []

    for i in y_pred:
        # standard distribution of values
        weigth_std.append(np.std(i))

        # entropie if this values corresbond to a normal distribution
        # weigth_entropy.append(scipy.stats.entropy(scipy.stats.norm.pdf(i, loc=0.5, scale=0.25)))
        # not really working, probability is hard to come by
        number_classes = Y.shape[-1]
        weigth_entropy.append(scipy.stats.entropy(scipy.stats.norm.pdf(i, loc=1 / number_classes, scale=0.5)))

    for i in range(0, int(len(y_pred) / number_subsequences)):
        sample_pred_mean = np.array(
            np.sum(y_pred[i * number_subsequences:i * number_subsequences + number_subsequences],
                   axis=0) / number_subsequences)
        y_pred_mean.append(np.argmax(sample_pred_mean))
        y_pred_mean_exact.append(sample_pred_mean)

        sample_weigths = weigth_entropy[i * number_subsequences:i * number_subsequences + number_subsequences]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        # add entropy weighted prediction
        y_pred_mean_weight_ent.append(np.argmax(np.array(
            np.sum(
                np.array(y_pred[i * number_subsequences:i * number_subsequences + number_subsequences]) * sw_normalized,
                axis=0) / number_subsequences)))

        sample_weigths = weigth_std[i * number_subsequences:i * number_subsequences + number_subsequences]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        # add standard deviation weighted prediction
        y_pred_mean_weight_std.append(np.argmax(np.array(
            np.sum(
                np.array(y_pred[i * number_subsequences:i * number_subsequences + number_subsequences]) * sw_normalized,
                axis=0) / number_subsequences)))

    # print(y_pred[i * number_subsequences:i * number_subsequences + number_subsequences])
    # standard respond
    y_pred = np.argmax(y_pred, axis=-1)

    # count all votes for one big sequence
    y_true = np.argmax(Y, axis=-1)
    y_true_small, y_pred_voted = [], []
    """calc voting of sequence (via votings for subsequences)"""
    for i in range(0, int(len(y_true) / number_subsequences)):
        arr = np.bincount(y_pred[i * number_subsequences:i * number_subsequences + number_subsequences])
        best = np.argwhere(arr == np.amax(arr)).flatten()
        y_pred_voted.append(np.random.permutation(best)[
                                0])  # if (very unlikely) multiple classes have the same vote prob. take randomly one

        y_true_small.append(np.argmax(
            np.array(np.bincount(y_true[i * number_subsequences:i * number_subsequences + number_subsequences]))))

    if do_print:
        print_predictions(y_true, y_pred, y_true_small, y_pred_voted, y_pred_mean, y_pred_mean_weight_std,
                          y_pred_mean_weight_ent)
    return y_true_small, y_pred_mean, y_pred_voted, y_pred, np.array(y_pred_mean_exact)


def run_tests_for_plotting(inpath):
    """
    setups for some of the tests from the masterthesis
    :return:
    """

    files = os.listdir(inpath)
    if "Y_train.csv" in files:
        general_setting = {"path": inpath, "suffix": "model7_deeperDense_ROC", "sampleSize": 1,
                           "do_shrink_timesteps": True, "voting": True, "design": 7, "nodes": 150, "faster": True,
                           "gpus": 1,
                           "titel": "Accuracy CNN biLSTM normal-repeat, TBTT", "accuracy": True,
                           "dropout": 0.2, "epochs": 500, "repeat": True, "use_repeat_spacer": False,
                           "batch_norm": False, "randomrepeat": False,
                           "online_training": False, "shuffleTraining": True, "use_generator": True}

        """online training"""
        test_setting = general_setting.copy()
        test_setting.update({"suffix": "online_design7", "online_training": True, "input_subSeqlength": 400})
        test_and_plot(**test_setting)

        """use no repeat but gaps"""
        test_setting = general_setting.copy()
        test_setting.update({"suffix": "no_repeat_but_gaps_design7", "repeat": False})
        test_and_plot(**test_setting)


def test_and_plot(inpath, outpath, suffix, batch_norm=False, filter_trainset=False, use_old_dataset=False,
                  do_shrink_timesteps=True, online_training=False, shuffleTraining=True, early_stopping_bool=True,
                  one_hot_encoding=True, repeat=True, use_repeat_spacer=False, randomrepeat=False, val_size=0.3,
                  input_subSeqlength=0, design=1, sampleSize=1, nodes=32, use_generator=True,
                  snapShotEnsemble=False, epochs=100, dropout=0, faster=False,
                  voting=False, tensorboard=False, gpus=False, titel='', x_axes='', y_axes='', accuracy=False,
                  loss=False, runtime=False, label1='', label2='', label3='', label4='', **kwargs):
    """
    1. gets settings and prepare data
    2. saves settings
    3. starts training
    4. saves history
    5. plots results
    :return:
    """

    # GET SETTINGS AND PREPARE DATA
    global X_train, X_test, X_val, Y_train, Y_test, Y_val, batch_size, SEED, y_encoder, number_subsequences

    # if use_old_dataset:
    # 	use_old_data(one_hot_encoding=one_hot_encoding)
    # else:
    Y_train_old = pd.read_csv(inpath + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    Y_train, y_encoder = DataParsing_main.encode_string(y=Y_train_old)
    print(*(zip(y_encoder.transform(y_encoder.classes_), y_encoder.classes_)))
    Y_train_noOHE = [y.argmax() for y in Y_train]
    class_weight = clw.compute_class_weight('balanced', np.unique(Y_train_noOHE), Y_train_noOHE)
    unbalanced = any([i != class_weight[0] for i in class_weight])
    maxLen = use_data_nanocomb(directory=inpath, one_hot_encoding=one_hot_encoding, repeat=repeat,
                               use_spacer=use_repeat_spacer,
                               online=online_training, unbalanced=unbalanced, maxLen=kwargs.get("maxLen", 0))

    if len(X_val) == 0:
        print("make new val set")
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=SEED,
                                                          stratify=Y_train)
    else:
        print("val set already exits")

    if nodes < X_test.shape[-1]:
        warning("use at least as many nodes as number of hosts to predict, better twice as much")

    if online_training == True:
        if do_shrink_timesteps == False:
            warning(
                "activate do_shrink_timesteps if you wanna use online training, we will now set do_shrink_timesteps=True")
            do_shrink_timesteps = True

    if do_shrink_timesteps:

        if not online_training:
            X_train, Y_train, number_subsequences = DataParsing_main.shrink_timesteps(X_train, Y_train,
                                                                                      input_subSeqlength)
        X_val, Y_val, number_subsequences = DataParsing_main.shrink_timesteps(X_val, Y_val, input_subSeqlength)
        X_test, Y_test, number_subsequences = DataParsing_main.shrink_timesteps(X_test, Y_test, input_subSeqlength)
    else:
        number_subsequences = 1

    print(f"number of subsequences used per sequence: {number_subsequences}")
    """to limit the training on specified classes/hosts"""
    if filter_trainset:
        filter_train_data()

    # SAVE SETTINGS
    with open(inpath + '/' + suffix + "_config.txt", "w") as file:
        for i in list(locals().items()):
            if i == 'Y_train_noOHE':
                continue

            file.write(str(i) + '\n')
        if faster == True:
            file.write('(\'batchsize\', ' + str(batch_size * 16) + ')\n')
        elif type(faster) == int and faster > 0:
            file.write('(\'batchsize\', ' + str(batch_size * faster) + ')\n')
        else:
            file.write('(\'batchsize\', ' + str(batch_size) + ')\n')
        file.write('(\'SEED\', ' + str(SEED) + ')\n')
        file.write('(\'directory\', ' + str(directory) + ')\n')

    # START TRAINING
    model_for_plot(inpath=inpath, outpath=outpath, design=design, sampleSize=sampleSize, nodes=nodes,
                   suffix=suffix, epochs=epochs, early_stopping_bool=early_stopping_bool,
                   dropout=dropout, use_generator=use_generator, faster=faster,
                   voting=voting, tensorboard=tensorboard, gpus=gpus,
                   snapShotEnsemble=snapShotEnsemble, shuffleTraining=shuffleTraining,
                   batch_norm=batch_norm, online_training=online_training,
                   do_shrink_timesteps=do_shrink_timesteps, repeat=repeat, use_spacer=use_repeat_spacer,
                   randomrepeat=randomrepeat, number_subsequences=number_subsequences, **kwargs)

    # model = result_dict["model"]
    # y_pred = model.predict(X_test)
    # calc_predictions(X_test, Y_test, y_pred, do_print=True)

    model_path1 = f"{inpath}/model_best_loss_{suffix}.hdf5"
    model_path2 = f"{inpath}/model_best_acc_{suffix}.hdf5"
    for model_path in (model_path1, model_path2):
        print("load model:")
        print(model_path)
        model = tf.keras.models.load_model(model_path)
        pred = model.predict(X_test)
        y_true_small, y_pred_mean, y_pred_voted, y_pred, y_pred_mean_exact = calc_predictions(X_test, Y_test,
                                                                                              y_pred=pred,
                                                                                              do_print=True)
        print("make test")
        myRoc = roc_History(name="_".join(model_path.split("_")[-3:-1]) + "_" + suffix, path=inpath)
        # myRoc = roc_History(name=suffix, path=inpath)
        myRoc.on_train_begin()
        global prediction_val
        prediction_val = model.predict(X_test)
        X_val = X_test
        Y_val = Y_test
        myRoc.on_epoch_end(0)

        # create and export .model file
        index_classes = dict()
        for i in zip(y_encoder.transform(y_encoder.classes_), y_encoder.classes_):
            index_classes.update({i[0]: i[1]})

        repeat = True
        use_spacer = False
        online = False
        random_repeat = True
        design = design
        multi_thresh = myRoc.thresholds[-1]
        hosts = Y_test.shape[-1]
        pickle.dump(
            (model.to_json(), model.get_weights(), index_classes, multi_thresh, maxLen, repeat, use_repeat_spacer,
             online_training, randomrepeat, design, hosts), open(f"{model_path.split('.')[0]}.model", "wb"))

    tf.keras.backend.clear_session()
    del model


"""some settings which are handy to have globally"""
X_test = []
X_val = []
X_train = []
Y_test = []
Y_val = []
Y_train = []
SEED = 42
batch_size = 32
y_encoder = None
directory = ''
number_subsequences = 1
# if __name__ == '__main__':
#     training()
