# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
# import random
# random.seed(3)
from keras.models import load_model
# import os
# os.environ['PYTHONHASHSEED'] = '0'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
# filters tensor flow output (the higher the number the more ist filtered)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # {0, 1, 2 (warnings), 3 (errors)}
import tensorflow as tf
import pickle
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, LSTM, Dense, Flatten, Embedding, Dropout, GRU, CuDNNLSTM, BatchNormalization, \
    Bidirectional, Conv1D, MaxPooling1D, Input, Concatenate, Activation, LeakyReLU
from .DataParsing import DataParsing
from keras.engine.saving import preprocess_weights_for_loading
import numpy as np
import pkg_resources

def get_model(design, X_test, hosts, nodes=150, dropout=0):
    timesteps = X_test.shape[1]
    model = Sequential()

    if design == 4:
        model.add(Bidirectional(LSTM(nodes, return_sequences=True, dropout=dropout),
                                input_shape=(timesteps, X_test.shape[-1])))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True, dropout=dropout)))
        model.add(Bidirectional(LSTM(nodes, dropout=dropout)))

    if design == 7:
        model.add(Conv1D(nodes, 9, input_shape=(timesteps, X_test.shape[-1])))
        # model.add(Activation('relu',alpha= 0.01))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(nodes, 9))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling1D(3))

        model.add(Bidirectional(LSTM(nodes, return_sequences=True, recurrent_activation="sigmoid"),
                                input_shape=(timesteps, X_test.shape[-1])))
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes, recurrent_activation="sigmoid")))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(dropout))
    # model.add(Dense((nodes+Y_train.shape[-1])//2, activation='elu'))
    # model.add(Dropout(dropout))
    model.add(Dense(hosts, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'], sample_weight_mode=None)
    return model


def gpu_weights_to_cpu_weights(model, cudnn_weights):
    i = 0
    weights2 = []
    for layer in model._layers:
        weight_len = len(layer.weights)
        if weight_len > 0:
            cudnn_weights_layer = cudnn_weights[i:i + weight_len]
            i += weight_len
            weights2_layer = preprocess_weights_for_loading(layer, cudnn_weights_layer)
            for j in range(len(weights2_layer)):
                weights2.append(weights2_layer[j])

    return weights2


def calc_predictions(batch_size, y_pred):
    """
    plot predictions
    :param X: raw-data which should be predicted
    :param Y: true labels for X
    :param do_print: True == print the cross-tab of the prediction
    :param y_pred: array with predicted labels for X
    :return: y_true_small == True labels for complete sequences, yTrue == True labels for complete subsequences, y_pred_mean == with mean predicted labels for complete sequences, y_pred_voted == voted labels for complete sequences, y_pred == predicted labels for complete subsequences
    """

    # for addition of probability and not voting
    y_pred_mean = []
    y_pred_mean_exact = []
    weigth_std = []
    y_pred_mean_weight_std = []
    y_pred_mean_weight_std_exact = []

    for i in y_pred:
        # standard distribution of values
        weigth_std.append(np.std(i))

    for i in range(0, int(len(y_pred) / batch_size)):
        sample_pred_mean = np.array(np.sum(y_pred[i * batch_size:i * batch_size + batch_size], axis=0) / batch_size)
        y_pred_mean.append(np.argmax(sample_pred_mean))
        y_pred_mean_exact.append(sample_pred_mean)

        sample_weigths = weigth_std[i * batch_size:i * batch_size + batch_size]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        # add standard deviation weighted prediction
        sample_pred_mean_weight_std = (np.array(
            np.sum(np.array(y_pred[i * batch_size:i * batch_size + batch_size]) * sw_normalized, axis=0) / batch_size))
        y_pred_mean_weight_std.append(np.argmax(sample_pred_mean_weight_std))
        y_pred_mean_weight_std_exact.append(sample_pred_mean_weight_std)

    # standard respond
    y_pred = np.argmax(y_pred, axis=-1)

    # count all votes for one big sequence
    y_pred_voted = []
    """calc voting of sequence (via votings for subsequences)"""
    for i in range(0, int(len(y_pred) / batch_size)):
        arr = np.array(np.bincount(y_pred[i * batch_size:i * batch_size + batch_size]))
        best = np.argwhere(arr == np.amax(arr)).flatten()
        y_pred_voted.append(np.random.permutation(best)[0])

    return y_pred_voted, y_pred_mean, y_pred_mean_weight_std, np.array(y_pred_mean_exact), np.array(
        y_pred_mean_weight_std_exact)


def show_output(index_classes, y_pred_mean_exact, y_pred_mean_weight_std_exact, top_n_host=False, threshold=False,
                filter=False):
    """using the Std-div version, if mean prefered comment next two lines"""
    # y_pred_mean_weight_std_exact_normed = y_pred_mean_weight_std_exact / np.sum(y_pred_mean_weight_std_exact)
    # y_pred_mean_exact = y_pred_mean_weight_std_exact_normed

    sorted_host_indices = np.argsort(y_pred_mean_exact)[::-1]

    if filter:
        print("per autofilter selected host of interest")
        sorted_filters = np.array(filter)[sorted_host_indices]
        sorted_host = y_pred_mean_exact[sorted_host_indices]
        filtered_host_bool = sorted_host > sorted_filters
        for index, boolean in enumerate(filtered_host_bool):
            if boolean == True:
                print(f"{index_classes[sorted_host_indices[index]]}: {sorted_host[index]}")
        if sum(filtered_host_bool) == 0:
            print(f"{index_classes[sorted_host_indices[0]]}: {sorted_host[0]}")
        print()

    if top_n_host:
        top_n_host_indices = sorted_host_indices[:top_n_host]
        print(f"top {top_n_host} host:")
        for host in top_n_host_indices:
            print(f"{index_classes[host]}: {y_pred_mean_exact[host]}")
        print()

    if threshold:
        print(f"all host with values over {threshold}")
        for host in sorted_host_indices:
            if y_pred_mean_exact[host] > threshold:
                print(f"{index_classes[host]}: {y_pred_mean_exact[host]}")
            else:
                break
        print()

    if not top_n_host and not threshold:
        print("all hosts")
        for host in sorted_host_indices:
            print(f"{index_classes[host]}: {y_pred_mean_exact[host]}")


def readDNA(path):
    with open(path) as f:
        dnaSeq = ""
        header_dict = {}
        header = None
        for line in f:
            if line.startswith(">"):
                if header != None:
                    header_dict.update({header: dnaSeq})
                dnaSeq = ""
                header = line[:-1]
            else:
                dnaSeq += line.upper()[:-1]
        header_dict.update({header: dnaSeq})
        return header_dict


def path_to_fastaFiles(mypath):
    """
    parse multiple fasta files to csv
    :param mypath: path to fasta files
    :param recursiv: if True use also sub-dirs
    :param n: length of the tuple's
    :return:
    """
    header_dict = {}
    if (os.path.isdir(mypath)):
        for root, dirs, files in os.walk(mypath):
            print(root)
            for filename in [f for f in files if f.endswith(".fna") or f.endswith(".fa") or f.endswith(".fasta")]:
                print(filename)
                data = str(os.path.join(root, filename))
                header_dict.update(readDNA(data))

    elif (os.path.isfile(mypath)):
        data = mypath
        header_dict.update(readDNA(data))
    else:
        assert set(mypath) <= set("ACGT"), "please input path to file or directory, or genome sequence"
        header_dict.update({">user command line input": mypath})
    return header_dict


def start_analyses(virus, top_n_host, threshold, X_test_old, header, auto_filter):

    assert len(X_test_old) >= 100, "sequence to short, please use sequences with at least 100 bases as input"

    if virus == "rota":
        maxLen = 3204
        repeat = True
        use_spacer = False
        online = False
        model_path = "/weights/rota_weights.best.loss.random_repeat_run1.pkl"
        random_repeat = True
        design = 4
        hosts = 6
        index_classes = {0: 'Bos taurus', 1: 'Equus caballus', 2: 'Gallus gallus', 3: 'Homo sapiens', 4: 'Sus scrofa',
                         5: 'Vicugna pacos'}
        a = "0.69 0.8 0.41 0.82 0.56 0.25"
        influ = a.split(" ")
        multi_thresh = [float(i) for i in influ]

    elif virus == "influ":
        maxLen = 2316
        repeat = True
        use_spacer = False
        online = True
        model_path = "/weights/influ_weights.best.acc.online_kickstart.pkl"
        random_repeat = False
        design = 4
        hosts = 49
        index_classes = {0: 'Anas acuta', 1: 'Anas carolinensis', 2: 'Anas clypeata', 3: 'Anas crecca',
                         4: 'Anas cyanoptera', 5: 'Anas discors', 6: 'Anas platyrhynchos', 7: 'Anas rubripes',
                         8: 'Anas sp.', 9: 'Anser albifrons', 10: 'Anser caerulescens', 11: 'Anser canagica',
                         12: 'Anser fabalis', 13: 'Anser indicus', 14: 'Anser sp.', 15: 'Arenaria interpres',
                         16: 'Aythya americana', 17: 'Aythya collaris', 18: 'Branta canadensis',
                         19: 'Bucephala albeola', 20: 'Cairina moschata', 21: 'Calidris alba', 22: 'Calidris canutus',
                         23: 'Calidris ruficollis', 24: 'Canis lupus', 25: 'Chroicocephalus ridibundus',
                         26: 'Clangula hyemalis', 27: 'Cygnus columbianus', 28: 'Cygnus cygnus', 29: 'Cygnus olor',
                         30: 'Equus caballus', 31: 'Gallus gallus', 32: 'Homo sapiens', 33: 'Larus argentatus',
                         34: 'Larus glaucescens', 35: 'Leiostomus xanthurus', 36: 'Leucophaeus atricilla',
                         37: 'Mareca americana', 38: 'Mareca penelope', 39: 'Mareca strepera',
                         40: 'Meleagris gallopavo', 41: 'Oxyura vittata', 42: 'Plantago princeps',
                         43: 'Sibirionetta formosa', 44: 'Struthio camelus', 45: 'Sus scrofa', 46: 'Tadorna ferruginea',
                         47: 'Turnix sp.', 48: 'Uria aalge'}
        a = "0.08 0.02 0.06 0.24 0.41 0.13 0.09 0.19 0.38 0.26 0.23 0.29 0.47 0.25 0.11 0.08 0.59 0.44 0.13 0.18 0.32" \
            " 0.29 0.17 0.63 0.21 0.79 0.23 0.27 0.48 0.41 0.61 0.31 0.45 0.17 0.41 0.35 0.15 0.33 0.19 0.19 0.14" \
            " 0.08 0.31 0.47 0.25 0.55 0.39 0.61 0.51"
        influ = a.split(" ")
        multi_thresh = [float(i) for i in influ]

    elif virus == "rabies":
        maxLen = 5052
        repeat = True
        use_spacer = False
        online = False
        random_repeat = True
        design = 4
        hosts = 19
        model_path = "/weights/rabies_weights.best.acc.random_repeat_run1.pkl"
        index_classes = {0: 'Artibeus lituratus', 1: 'Bos taurus', 2: 'Canis lupus', 3: 'Capra hircus',
                         4: 'Cerdocyon thous', 5: 'Desmodus rotundus', 6: 'Eptesicus fuscus', 7: 'Equus caballus',
                         8: 'Felis catus', 9: 'Homo sapiens', 10: 'Lasiurus borealis', 11: 'Mephitis mephitis',
                         12: 'Nyctereutes procyonoides', 13: 'Otocyon megalotis', 14: 'Procyon lotor',
                         15: 'Stomatepia mongo', 16: 'Tadarida brasiliensis', 17: 'Vulpes lagopus', 18: 'Vulpes vulpes'}
        a = "0.55 0.2 0.31 0.22 0.36 0.52 0.95 0.3 0.25 0.16 0.61 0.6 0.42 0.65 0.75 0.4 0.79 0.79 0.45"
        influ = a.split(" ")
        multi_thresh = [float(i) for i in influ]

    if not auto_filter:
        multi_thresh = False

    assert type(X_test_old)==str, "only prediction of single sequences per start_analyses instanz implemented yet"

    """due to random repeat make 10 predictions per sample"""
    X_test_old = [X_test_old] * 10

    """parse input"""
    X_test = DataParsing.encode_string(maxLen=maxLen, x=X_test_old, repeat=repeat, use_spacer=use_spacer,
                                       online_Xtrain_set=False, randomrepeat=random_repeat)

    X_test, Y_test, batch_size = DataParsing.shrink_timesteps(X_test, [], 0)

    """build model"""
    model = get_model(design=design, X_test=X_test, hosts=hosts)

    """load previously trained weights"""
    exact_path_to_model_weigth = pkg_resources.resource_filename("vidhop", model_path)
    weights = pickle.load(open(f"{exact_path_to_model_weigth}", "rb"))
    weights = gpu_weights_to_cpu_weights(model, weights)
    model.set_weights(weights)

    """predict input"""
    y_pred = model.predict_proba(X_test)

    """output modification"""
    y_pred_voted, y_pred_mean, y_pred_mean_weight_std, y_pred_mean_exact, y_pred_mean_weight_std_exact = calc_predictions(
        batch_size, y_pred)

    """join 10 predictions"""
    y_pred_mean_exact = np.array([y_pred_mean_exact.mean(axis=0)])
    y_pred_mean_weight_std_exact = np.array([y_pred_mean_weight_std_exact.mean(axis=0)])

    for sample_number in range(0, y_pred_mean_exact.shape[0]):
        print(header)
        show_output(index_classes, y_pred_mean_exact[sample_number], y_pred_mean_weight_std_exact[sample_number],
                    top_n_host=top_n_host, threshold=threshold, filter=multi_thresh)


if __name__ == '__main__':
    virus = "rabies"
    top_n_host = False
    threshold = False
    # top_n_host = 3
    # threshold = 0.2
    X_test_old = "ACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTAACCCAATTACT"
    # X_test_old = [
    #     "GGGAAAGGGACATTTGAAAGAAGATTCTTCAGAGATGAGAAAGAACTTCAAGAATACGAGGCGGCTGAACTGACAAAGACTGACGTAGCACTGGCAGATGATGGAACTGTCAACTCTGACGACGAGGACTACTTCTCAGGTGAAACCAGAAGTCCGGAAGCTGTTTATACTCGAATCATAATGAATGGAGGTCGACTGAAGAGATCGCACATACGGAGATATGTCTCAGTCAGTTCCAATCATCAAGCTCGTCCAAACTCATTCGCCGAGTTTCTAAACAAGACATATTCGAGTGACTCATAAGAAGTTGAATAACAAAA",
    #     "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTAACCCAATTAACCCAATTAACCCAATTAACTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCCAATTACT"]

    start_analyses(virus="influ", top_n_host=top_n_host, threshold=threshold, X_test_old=X_test_old, header=">test",
                   auto_filter=True)
