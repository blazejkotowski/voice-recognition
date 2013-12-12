#!/usr/bin/env python
import scipy as sp
from scipy.io import wavfile as wav
import numpy as np
import os
from pylab import *

files = map(lambda x: 'train/' + x, os.listdir('train/'))

if __name__ == '__main__':
  for filename1, filename2 in zip(files[2:], files[3:]):
    signal1 = np.fromfile(open(filename1),np.int16)
    signal1 = signal1[12:len(signal1)]

    signal2 = np.fromfile(open(filename2),np.int16)
    signal2 = signal2[12:len(signal2)]

    freqs1 = abs(sp.fft(signal1))
    freqs2 = abs(sp.fft(signal2))
    # args = linspace(0, maxfreqs, , endpoint=False)

    subplot(211)
    plot(freqs1[1:len(freqs1)/2])
    xlim(0, 23000)
    ylabel(filename1)
    # yscale('log')
    # xscale('log')
    # stem(args, freqs, '-*')
    subplot(212)
    plot(freqs2[1:len(freqs2)/2])
    ylabel(filename2)
    xlim(0, 23000)
    # yscale('log')
    show()


    break

    
  
