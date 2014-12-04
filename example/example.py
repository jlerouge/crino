#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright (c) 2014 Clément Chatelain, Romain Hérault, Julien Lerouge,
#    Romain Modzelewski (LITIS - EA 4108). All rights reserved.
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

import matplotlib
matplotlib.use('pdf')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

import crino
from crino.criterion import MeanSquareError

# If learn is true, the example will learn a IODA network from training data
# Else, it will load a IODA network saved from a previous run
learn = True # False

# Learning parameters
if(learn):
    input_pretraining_params={
         'learning_rate': 10.0,
         'batch_size' : 100,
         'epochs' : 3
         }
    output_pretraining_params={
         'learning_rate': 10.0,
         'batch_size' : 100,
         'epochs' : 3
         }    
    link_pretraining_params={
         'learning_rate': 10.0,
         'batch_size' : 100,
         'epochs' : 3
         }  
    learning_params={
        'learning_rate' : 2.0,
        'batch_size' : 100,
        'epochs' : 3,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        'link_pretraining_params' : link_pretraining_params,
        'link_pretraining' : True
        
    }
    
    hidden_size = 10
    inputlayers=[hidden_size, hidden_size]
    #inputlayers=[]
    outputlayers=[hidden_size, hidden_size]
    #outputlayers=[]
    linklayers=[hidden_size, hidden_size]
    #linklayers=[]
print '... loading training data'
train_set = sio.loadmat('data/train.mat')
x_train = np.asarray(train_set['x_train'], dtype=theano.config.floatX) # We convert to float32 to 
y_train = np.asarray(train_set['y_train'], dtype=theano.config.floatX) # compute on GPUs with CUDA
N = x_train.shape[0] # number of training examples
nFeats = x_train.shape[1] # number of pixels per image
xSize = int(np.sqrt(nFeats)) # with of a square image
print("Image of size %d x %d"%(xSize,xSize))


# Construct a IODA network on training data
if(learn):
    print '... building and learning a IODA network'
    nn = crino.network.DeepNeuralNetwork(nFeats, nFeats, crino.module.Sigmoid, nUnitsInput=inputlayers, nUnitsLink=linklayers, nUnitsOutput=outputlayers)
    nn.linkInputs(T.matrix('x'), nFeats)
    nn.prepare()
    nn.criterion = MeanSquareError(nn.outputs, T.matrix('y'))
    delta = nn.train(x_train, y_train, **learning_params)
    print '... learning lasted %s (s) ' % (delta)
    print '... saving the IODA network to data/ioda.nn'
    nn.save('data/ioda.nn')
else:
    print '... loading the existing IODA network from data/ioda.nn'
    nn = crino.module.load('data/ioda.nn')

print '... loading test data'
test_set = sio.loadmat('data/test.mat')
x_test = np.asarray(test_set['x_test'], dtype=theano.config.floatX) # We convert to float32 to
y_test = np.asarray(test_set['y_test'], dtype=theano.config.floatX) # compute on GPUs with CUDA
N = x_test.shape[0] # number of test examples

print '... applying the learned IODA network on test data'
plt.close('all')
y_estim_full = nn.forward(x_test)

for k in xrange(N):
    print("Testing %d/%d"%(k+1,N))
    x_orig = np.reshape(x_test[k:k+1], (xSize, xSize), 'F')
    y_true = np.reshape(y_test[k:k+1], (xSize, xSize), 'F')
    y_estim = np.reshape(y_estim_full[k:k+1], (xSize, xSize), 'F')

    # Plot the results
    plt.figure(k)
    plt.subplot(2,2,1)
    plt.imshow(x_orig, interpolation='bilinear', cmap=cm.gray)
    plt.title('Original input')
    plt.subplot(2,2,2)
    plt.imshow(y_true, interpolation='bilinear', cmap=cm.gray)
    plt.title('Target')
    plt.subplot(2,2,3)
    plt.imshow(y_estim, interpolation='bilinear', cmap=cm.gray)
    plt.title('Estimated output')
    plt.savefig("figure/ioda_%03d.pdf"%(k,))
    plt.close()
