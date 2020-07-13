import re

import numpy as np
from sklearn.preprocessing import LabelEncoder

seed = 42
import random
random.seed(seed)
from tensorflow.keras.utils import to_categorical
from logging import warning
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import operator


class CircularList(list):
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)


def encode_string(maxLen=None, x=[], y=[], y_encoder=None, repeat=True, use_spacer=False, online_Xtrain_set=False,
                  randomrepeat=False):
    """
    One hot encoding for classes
    to convert the "old" exported int data via OHE to binary matrix
    http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    for dna ony to int values
    """

    def pad_n_repeat_sequences(sequences, maxlen=None, dtype='int32',
                               padding='post', truncating='post', value=0.):
        """extended version of pad_sequences()"""
        if not hasattr(sequences, '__len__'):
            raise ValueError('`sequences` must be iterable.')
        lengths = []
        for x in sequences:
            if not hasattr(x, '__len__'):
                raise ValueError('`sequences` must be a list of iterables. '
                                 'Found non-iterable: ' + str(x))
            lengths.append(len(x))
        num_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        # make new array and fill with input seqs
        x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError(
                    'Shape of sample %s of sequence at position %s is different from expected shape %s' %
                    (trunc.shape[1:], idx, sample_shape))

            if repeat:
                # repeat seq multiple times
                repeat_seq = np.array([], dtype=dtype)
                while len(repeat_seq) < maxLen:
                    if use_spacer:
                        spacer_length = random.randint(1, 50)
                        spacer = [value for i in range(spacer_length)]
                        repeat_seq = np.append(repeat_seq, spacer)
                        if randomrepeat:
                            random_start = random.randint(0, len(trunc))
                            repeat_seq = np.append(repeat_seq,
                                                   CircularList(trunc)[random_start:random_start + len(trunc)])
                        else:
                            repeat_seq = np.append(repeat_seq, trunc)
                    else:
                        if randomrepeat:
                            random_start = random.randint(0, len(trunc))
                            repeat_seq = np.append(repeat_seq,
                                                   CircularList(trunc)[random_start:random_start + len(trunc)])
                        else:
                            repeat_seq = np.append(repeat_seq, trunc)
                x[idx, :] = repeat_seq[-maxLen:]

            else:
                if padding == 'post':
                    x[idx, :len(trunc)] = trunc
                elif padding == 'pre':
                    x[idx, -len(trunc):] = trunc
                else:
                    raise ValueError('Padding type "%s" not understood' % padding)

        return x

    encoder = LabelEncoder()

    if len(x) > 0:
        a = "ATGCN-"

        encoder.fit(list(a))
        out = []
        if type(x)==str:
            dnaSeq = re.sub(r"[^ACGTUacgtu]", 'N', x)
            encoded_X = encoder.transform(list(dnaSeq))
            out.append(encoded_X)
        else:
            for i in x:
                dnaSeq = re.sub(r"[^ACGTUacgtu]", 'N', i)
                # dnaSeq = i[0]
                encoded_X = encoder.transform(list(dnaSeq))
                out.append(encoded_X)

        if online_Xtrain_set:
            X_train_categorial = []
            for seq in out:
                X_train_categorial.append(np.array(to_categorical(seq, num_classes=len(a)), dtype=np.bool))
            return X_train_categorial
        else:
            out = pad_n_repeat_sequences(out, maxlen=maxLen, dtype='int16', truncating='pre', value=0)

        return np.array(to_categorical(out, num_classes=len(a)), dtype=np.bool)
    else:
        if y_encoder != None:
            encoder.fit(y)
            if np.array(encoder.classes_ != y_encoder.classes_).all():
                warning(f"Warning not same classes in training and test set")
            useable_classes = set(encoder.classes_).intersection(y_encoder.classes_)
            try:
                assert np.array(encoder.classes_ == y_encoder.classes_).all()
            except AssertionError:
                warning(
                    f"not all test classes in training data, only {useable_classes} predictable "
                    f"from {len(encoder.classes_)} different classes\ntest set will be filtered so only predictable"
                    f" classes are included")

            try:
                assert len(useable_classes) == len(encoder.classes_)
            except AssertionError:
                print(f"not all test classes in training data, only " \
                      f"{useable_classes} predictable from {len(encoder.classes_)} different classes" \
                      f"\ntest set will be filtered so only predictable classes are included")

            if not len(useable_classes) == len(encoder.classes_):
                global X_test, Y_test
                arr = np.zeros(X_test.shape[0], dtype=int)
                for i in useable_classes:
                    arr[y == i] = 1

                X_test = X_test[arr == 1, :]
                y = y[arr == 1]
                encoded_Y = y_encoder.transform(y)
            else:
                encoded_Y = encoder.transform(y)

            return to_categorical(encoded_Y, num_classes=len(y_encoder.classes_))

        else:
            encoder.fit(y)
            # print(encoder.classes_)
            # print(encoder.transform(encoder.classes_))

            encoded_Y = encoder.transform(y)
            return to_categorical(encoded_Y), encoder


def manipulate_training_data(X, Y, subSeqLength, number_subsequences):
    pool = ThreadPool(multiprocessing.cpu_count())

    def make_manipulation(sample):
        if len(sample) >= subSeqLength:
            X_train_manipulated = []
            # sample_long = sample.tolist() * number_subsequences
            for i in range(number_subsequences):
                start = random.randint(0, len(sample) - subSeqLength)
                subSeq = sample[start:start + subSeqLength]
                X_train_manipulated.append(subSeq)
            # i = 0
            # while i < number_subsequences:
            #     X_train_manipulated.append(sample_long[i*subSeqLength:i*subSeqLength+subSeqLength])
            #     i += 1

            return np.array(X_train_manipulated)
        else:
            return

    # X_train_manipulated_total = list(map(make_manipulation, X))
    X_train_manipulated_total = pool.map(make_manipulation, X)
    pool.close()
    pool.join()
    X_train_manipulated_total = np.array(X_train_manipulated_total)
    shape = X_train_manipulated_total.shape
    X_train_manipulated_total = X_train_manipulated_total.reshape(
        (len(X) * number_subsequences, shape[2], shape[3]))

    y = []
    for i in Y:
        y.append(number_subsequences * [i])

    Y = np.array(y)
    if len(Y.shape) == 2:
        Y = np.array(y).flatten()
    elif len(Y.shape) == 3:
        Y = Y.reshape((Y.shape[0] * Y.shape[1], Y.shape[2]))

    return X_train_manipulated_total, Y


def calc_shrink_size(seqlength):
    subSeqlength = 100
    for i in range(100, 400):
        if (seqlength % i == 0):
            subSeqlength = i

    batch_size = int(seqlength / subSeqlength)
    return subSeqlength, batch_size


def shrink_timesteps(X, Y, input_subSeqlength=0):
    """
        needed for Truncated Backpropagation Through Time
    If you have long input sequences, such as thousands of timesteps,
    you may need to break the long input sequences into multiple contiguous subsequences.

    e.g. 100 subseq.
    Care would be needed to preserve state across each 100 subsequences and reset
    the internal state after each 100 samples either explicitly or by using a batch size of 100.
    :param input_subSeqlength: set for specific subsequence length
    :return:
    """
    # assert input_subSeqlength != 0, "must provide variable \"input_subSeqlength\" when using shrink_timesteps for specific subset"
    if len(X.shape) == 3:
        seqlength = X.shape[1]
        features = X.shape[-1]

        if input_subSeqlength == 0:
            subSeqlength, batch_size = calc_shrink_size(seqlength)
        else:
            subSeqlength = input_subSeqlength
            batch_size = int(seqlength / subSeqlength)

        newSeqlength = int(seqlength / subSeqlength) * subSeqlength

        bigarray = []
        for sample in X:
            sample = np.array(sample[0:newSeqlength], dtype=np.bool)
            subarray = sample.reshape((int(seqlength / subSeqlength), subSeqlength, features))
            bigarray.append(subarray)
        bigarray = np.array(bigarray)
        X = bigarray.reshape((bigarray.shape[0] * bigarray.shape[1], bigarray.shape[2], bigarray.shape[3]))

    elif len(X.shape) == 2:
        seqlength = X.shape[0]
        features = X.shape[-1]

        if input_subSeqlength == 0:
            subSeqlength, batch_size = calc_shrink_size(seqlength)
        else:
            subSeqlength = input_subSeqlength
            batch_size = int(seqlength / subSeqlength)

        newSeqlength = int(seqlength / subSeqlength) * subSeqlength

        sample = np.array(X[0:newSeqlength], dtype=np.bool)
        subarray = sample.reshape((int(seqlength / subSeqlength), subSeqlength, features))
        X = np.array(subarray)

    else:
        assert len(X.shape) == 2 or len(
            X.shape) == 3, f"wrong shape of input X, expect len(shape) to be 2 or 3 but is instead {len(X.shape)}"
    y = []
    for i in Y:
        y.append(int(seqlength / subSeqlength) * [i])

    Y = np.array(y)
    if len(Y.shape) == 2:
        Y = np.array(y).flatten()
    elif len(Y.shape) == 3:
        Y = Y.reshape((Y.shape[0] * Y.shape[1], Y.shape[2]))

    return X, Y, batch_size
