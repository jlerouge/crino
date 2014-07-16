#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright (c) 2014 Clément Chatelain, Romain Hérault, Benjamin Labbé, Julien Lerouge,
#    Romain Modzelewski, LITIS - EA 4108. All rights reserved.
#    
#    This file is part of Crino.
#
#    Crino is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Crino is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with Crino. If not, see <http://www.gnu.org/licenses/>.

import itertools as it
import numpy as np
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import theano.tensor as T

import crino
from crino.criterion import MeanSquareError

# If learn is true, the example will learn a IODA network from training data
# Else, it will load a IODA network saved from a previous run
learn = True # False

# Learning parameters
if(learn):
    learning_rate = 2.0
    pretraining_learning_rate = 10.0
    minibatch_size = 100
    epochs = 300

print '... loading training data'
train_set = sio.loadmat('data/train.mat')
x_train = np.asarray(train_set['x_train'], dtype='float32') # We convert to float32 to 
y_train = np.asarray(train_set['y_train'], dtype='float32') # compute on GPUs with CUDA
N = x_train.shape[0] # number of training examples
nFeats = x_train.shape[1] # number of pixels per image
xSize = int(np.sqrt(nFeats)) # with of a square image

# Construct a IODA network on training data
if(learn):
    print '... building and learning a IODA network'
    nn = crino.network.InputOutputDeepArchitecture([nFeats, xSize*8], [xSize*8, nFeats], crino.module.Sigmoid)
    nn.linkInputs(T.matrix('x'), nFeats)
    nn.prepare()
    nn.criterion = MeanSquareError(nn.outputs, T.matrix('y'))
    delta = nn.train(x_train, y_train, minibatch_size, learning_rate, pretraining_learning_rate, epochs, verbose=True)
    print '... learning lasted %f minutes ' % (delta / 60.)
    print '... saving the IODA network to data/ioda.nn'
    nn.save('data/ioda.nn')
else:
    print '... loading the existing IODA network from data/ioda.nn'
    nn = crino.module.load('data/ioda.nn')

print '... loading test data'
test_set = sio.loadmat('data/test.mat')
x_test = np.asarray(test_set['x_test'], dtype='float32') # We convert to float32 to
y_test = np.asarray(test_set['y_test'], dtype='float32') # compute on GPUs with CUDA
N = x_test.shape[0] # number of test examples

print '... applying the learned IODA network on test data'
for k in xrange(N):
    x_orig = np.reshape(x_test[k:k+1], (xSize, xSize), 'F')
    y_true = np.reshape(y_test[k:k+1], (xSize, xSize), 'F')
    y_estim = nn.forward(x_test[k:k+1])
    y_estim = np.reshape(y_estim, (xSize, xSize), 'F')

    # Plot the results
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(x_orig, interpolation='bilinear', cmap=cm.gray)
    plt.title('Original input')
    plt.subplot(2,2,2)
    plt.imshow(y_true, interpolation='bilinear', cmap=cm.gray)
    plt.title('Target')
    plt.subplot(2,2,3)
    plt.imshow(y_estim, interpolation='bilinear', cmap=cm.gray)
    plt.title('Estimated output')
    plt.show()
