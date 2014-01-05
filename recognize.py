#!/usr/bin/env python
from scikits.audiolab import wavread
import numpy as np
import os
from os.path import isfile
from math import ceil
import cPickle as pickle
import sys

import matplotlib.pyplot as mpl
from scipy.signal import decimate
from sklearn import tree

def samples(directory):
    for filename in map(lambda _: directory + _, os.listdir(directory)):
        signal, sample_frequency, _ = wavread(filename)

        if signal.ndim > 1:
            transposed = signal.transpose()
            signal = (transposed[0] + transposed[1])/2.0

        yield (os.path.basename(filename)[4], signal, sample_frequency)

def normalized_fft(signal, freq):
    spectrum = np.abs(np.fft.rfft(signal))/signal.size
    bins = np.fft.fftfreq(signal.size, 1.0/freq)

    return (spectrum, bins)

def aggregate_freqs(spectrum, bins, max_freq=6000, freq_step=10):
    new_spectrum = np.zeros(ceil(float(max_freq) / freq_step)+1)
    new_bins = range(0, max_freq+1, freq_step)

    for bar, center in zip(spectrum, bins):
        if ceil(center) > max_freq:
            break
        if int(ceil(center)/freq_step) <= 1:
            continue
        new_spectrum[int(ceil(center)/freq_step)] += bar


    return (new_spectrum, new_bins)

def train_classifier(directory):
    clf = tree.DecisionTreeClassifier()

    X, Y = [], []
    for sample in samples(directory):
        spectrum, bins = aggregate_freqs(*normalized_fft(*sample[1:]),
                freq_step = 50)
        X += [list(spectrum)]
        Y += [1 if sample[0] == 'M' else 0]
    clf.fit(X, Y)

    return clf

if __name__ == '__main__':
    if isfile("gender.clf"):
        print 'Using previously trained classifier.'
        print 'Remove gender.clf to train new one.'
        clf = pickle.load(open('gender.clf', 'rb'))
    else:
        print "Training new classifier."
        clf = train_classifier('train/')
        pickle.dump(clf, open('gender.clf', 'wb'))

    test_directory = 'train/'
    if len(sys.argv) > 1:
        test_directory = sys.argv[1]

    number_of_samples = 0
    success_count = 0
    for sample in samples(test_directory):
        spectrum, bins = aggregate_freqs(*normalized_fft(*sample[1:]),
                freq_step = 50)

        cls = clf.predict([list(spectrum)])

        number_of_samples += 1
        if sample[0] == ('M' if cls[0] == 1 else 'K'):
            success_count += 1

        print "Should be %s, recognized as %s" % (sample[0], 'M' if cls[0] == 1 else 'K')

    print "Recognized %i out of %i samples, efficiency %.2f%%" % (success_count,
            number_of_samples, float(success_count)/number_of_samples*100)

