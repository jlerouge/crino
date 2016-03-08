#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright (c) 2014-2015 Soufiane Belharbi, Clément Chatelain,
#    Romain Hérault, Julien Lerouge, Romain Modzelewski (LITIS - EA 4108).
#    All rights reserved.
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

import numpy as np

import sys

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import crino
from crino.criterion import NegativeLogLikelihood

learning_params= {
    'learning_rate' : 0.5,
    'batch_size' : 100,
    'epochs' : 10
}

print '... generating training data'
train_set = {}
napps=[100,100,100,100];
ntests=[30,30,30,30]
mus=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
s=0.2
sigmas=[np.eye(2)*s,np.eye(2)*s,np.eye(2)*s,np.eye(2)*s]
nClasses=4

x_train=np.zeros((0,2))
y_train=np.zeros((0,nClasses))
x_test=np.zeros((0,2))
y_test=np.zeros((0,nClasses))

for c,(mu,sigma,napp,ntest) in enumerate(zip(mus,sigmas,napps,ntests)):
    x_train=np.vstack([x_train,np.random.multivariate_normal(mu,sigma,napp)])
    y=np.zeros((napp,nClasses))
    y[:,c]=1
    y_train=np.vstack([y_train,y])
    x_test=np.vstack([x_test,np.random.multivariate_normal(mu,sigma,ntest)])
    y=np.zeros((ntest,nClasses))
    y[:,c]=1
    y_test=np.vstack([y_test,y])

arrtrain = np.arange(x_train.shape[0])
x_train=x_train[arrtrain]
y_train=y_train[arrtrain]

arrtest = np.arange(x_test.shape[0])
x_test=x_test[arrtest]
y_test=y_test[arrtest]

x_train = np.asarray(x_train, dtype=theano.config.floatX) # We convert to float32 to
y_train = np.asarray(y_train, dtype=theano.config.floatX) # compute on GPUs with CUDA
x_test = np.asarray(x_test, dtype=theano.config.floatX) # We convert to float32 to
y_test = np.asarray(y_test, dtype=theano.config.floatX) # compute on GPUs with CUDA

N = x_train.shape[0] # number of training examples
nFeats = x_train.shape[1] # number of features per example

print '... building and learning a MLP network'
nn = crino.network.MultiLayerPerceptron([2,4,nClasses] ,crino.module.Softmax)
nn.setInputs(T.matrix('x'), nFeats)
nn.prepare()
nn.setCriterion(NegativeLogLikelihood(nn.getOutputs(), T.matrix('y')))

delta = nn.train(x_train, y_train, **learning_params)
print '... learning lasted %s ' % (delta)



print '... testing'
y_test_estim = nn.forward(x_test)
y_test_estim=y_test_estim.argmax(axis=1)

# Plot the results
styles=['r+','g+','b+','k+']
plt.close()
plt.figure(1)
for c in xrange(nClasses):
    idx=np.nonzero(y_test_estim==c)[0]
    plt.plot(x_test[idx,0],x_test[idx,1],styles[c])
plt.savefig(sys.argv[0]+'.pdf')
plt.close()
