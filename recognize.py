#!/usr/bin/env python
import scipy as sp
from scipy.io import wavfile as wav
import numpy as np
import os
from pylab import *

files = map(lambda x: 'train/' + x, os.listdir('train/'))

if __name__ == '__main__':
  for filename in files:
    print filename
    _,signal = wav.read(filename)
    sp.fft(signal)
    subplot(211)
    args = linspace(0, len(signal), len(signal), endpoint=False)
    stem(args, signal, '-*')

    break

    
  
