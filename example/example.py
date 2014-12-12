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

import os,os.path
import sys

import numpy as np

import scipy
import scipy.io as sio

import theano
import theano.tensor as T

import crino
from crino.network import PretrainedMLP
from crino.criterion import MeanSquareError

import cPickle as pickle

import csv

input_pretraining_params={
        'learning_rate': 10.0,
        'batch_size' : 100,
        'epochs' : 300
        }
output_pretraining_params={
        'learning_rate': 10.0,
        'batch_size' : 100,
        'epochs' : 300
        }    
#link_pretraining_params={
        #'learning_rate': 1.0,
        #'batch_size' : 100,
        #'epochs' : 10
        #}  
learning_params={
    'learning_rate' : 2.0,
    'batch_size' : 100,
    'epochs' : 300,
    'input_pretraining_params' : input_pretraining_params,
    'output_pretraining_params' : output_pretraining_params,
    #'link_pretraining_params' : link_pretraining_params,
    'link_pretraining' : False  
}
    
hidden_size = 1024

outfolder='./results/'

exemples=[10,50,100]
displayepochs=[0,10,100,200,300]


class MyPretrainedMLP(PretrainedMLP):
    def setTestSet(self,x_test,y_test):
        self.shared_x_test=theano.shared(x_test)
        self.shared_y_test=theano.shared(y_test)
        
    def initEpochHook(self):
        self.testCriterionFunction=self.criterionFunction(downcast=True, shared_x_data=self.shared_x_test, shared_y_data=self.shared_y_test)
        self.testForwardFunction=self.forwardFunction(downcast=True, shared_x_data=self.shared_x_test)
        
        self.test_criterion_history=[np.mean(self.testCriterionFunction())]
        self.test_forward_history=[(-1,self.testForwardFunction())]
        
        self.appForwardFunction=self.forwardFunction(downcast=True, shared_x_data=self.finetunevars['shared_x_train'])
        self.app_forward_history=[(-1,self.appForwardFunction())]
        
    def checkEpochHook(self):
        self.test_criterion_history.append(np.mean(self.testCriterionFunction()))
        if self.finetunevars['epoch']+1 in displayepochs:
            self.test_forward_history.append((self.finetunevars['epoch'],self.testForwardFunction()))
            self.app_forward_history.append((self.finetunevars['epoch'],self.appForwardFunction()))


def data2greyimg(filename, X):
    Xn=(X-X.min())/(X.max()-X.min())*255
    scipy.misc.imsave(filename, Xn)

def main():   
    print '... loading training data'
    train_set = sio.loadmat('data/fixed/train.mat')
    x_train = np.asarray(train_set['x_train'], dtype=theano.config.floatX) # We convert to float32 to 
    y_train = np.asarray(train_set['y_train'], dtype=theano.config.floatX) # compute on GPUs with CUDA

    print '... loading test data'
    test_set = sio.loadmat('data/fixed/test.mat')
    x_test = np.asarray(test_set['x_test'], dtype=theano.config.floatX) # We convert to float32 to
    y_test = np.asarray(test_set['y_test'], dtype=theano.config.floatX) # compute on GPUs with CUDA


    nApp = x_train.shape[0] # number of training examples
    nTest = x_test.shape[0] # number of training examples
    nFeats = x_train.shape[1] # number of pixels per image
    xSize = int(np.sqrt(nFeats)) # with of a square image

    nInputs=nFeats
    nOutputs=nFeats
    
    geometry=[nFeats,hidden_size,hidden_size,nFeats]
    nLayers=len(geometry)-1

    # All configurations have the same geometry 3 layers and 4 representations
    # with sizes [nFeats,hidden_size,hidden_size,nFeats].
    # They only differ in the way layers are pretrained or not.
    configurations=[]
    # Standard MLP, no pretraining
    configurations.append({'nInputLayers':0,'nOutputLayers':0})
    # First layer pretrained input way
    configurations.append({'nInputLayers':1,'nOutputLayers':0})
    # Two first layer pretrained input way
    configurations.append({'nInputLayers':2,'nOutputLayers':0})
    # Last layer pretrained output way
    configurations.append({'nInputLayers':0,'nOutputLayers':1})
    # Two last layer pretrained output way
    configurations.append({'nInputLayers':0,'nOutputLayers':2})
    # First layer pretrained input way, and last layer pretrained output way
    configurations.append({'nInputLayers':1,'nOutputLayers':1})

    parameters=None
    #We throw random parameters only for the first conf and then reuse the same parameters for the remaining confs.

    absoutfolder=os.path.abspath(outfolder)
    if not os.path.exists(absoutfolder):
        os.mkdir(absoutfolder)

    results={}
  
    for phase,xdata,ydata in [['train',x_train,y_train],['test',x_test,y_test]]:  
        for ex in exemples:
            x_orig = np.reshape(xdata[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_input.png"%(phase,ex,)),x_orig)
            y_true = np.reshape(ydata[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_target.png"%(phase,ex,)),y_true)

    for conf in configurations:
        
        expname="I_%d_L_%d_O_%d"%(conf['nInputLayers'],nLayers-conf['nInputLayers']-conf['nOutputLayers'],conf['nOutputLayers'])
        print '... building and learning a network %s'%(expname,)
        nn = MyPretrainedMLP(geometry, outputActivation=crino.module.Sigmoid,**conf)
        nn.setTestSet(x_test,y_test)
        nn.linkInputs(T.matrix('x'), nFeats)
        nn.prepare()
        nn.criterion = MeanSquareError(nn.outputs, T.matrix('y'))
        if parameters is None:
            parameters=nn.getParameters()
            pickle.dump(parameters,open(os.path.join(absoutfolder,"starting_params.pck"),'w'),protocol=-1)
        else:
            nn.setParameters(parameters)
        delta = nn.train(x_train, y_train, **learning_params)
        print '... learning lasted %s (s) ' % (delta)
        
        results[expname]={
            'I':conf['nInputLayers'],
            'L':nLayers-conf['nInputLayers']-conf['nOutputLayers'],
            'O':conf['nOutputLayers'],
            'train_criterion':nn.finetunevars['history'][-1],
            'train_history':nn.finetunevars['history'],
            'train_full_history':nn.finetunevars['full_history'],
            'test_criterion': nn.test_criterion_history[-1],
            'test_history':nn.test_criterion_history,
            }
        pickle.dump(nn.getParameters(),open(os.path.join(absoutfolder,"%s_params.pck"%(expname,)),'w'),protocol=-1)
        
        for phase,xdata,ydata,history in [
                    ['train',x_train,y_train,nn.app_forward_history],
                    ['test',x_test,y_test,nn.test_forward_history]]:
            for ex in exemples:
                for epoch,forward in history:
                    y_estim = np.reshape(forward[ex:ex+1], (xSize, xSize), 'F')
                    data2greyimg(os.path.join(absoutfolder,"%s_%s_ex_%03d_estim_%03d.png"%(expname,phase,ex,epoch+1)),y_estim)
                    
    pickle.dump(results,open(os.path.join(absoutfolder,'results.pck'),'w'),protocol=-1)  
    
    table=[["Input Pretrained Layers","Link Layers","Output Pretrained Layers", "Epoch","Train", "Test"]]
    for expname in results.keys():
        for epoch in displayepochs:
            table.append([results[expname]['I'],results[expname]['L'],results[expname]['O'],epoch,results[expname]['train_history'][epoch],results[expname]['test_history'][epoch]])

    writer=csv.writer(open(os.path.join(absoutfolder,'results.csv'),'wb'),delimiter='\t')
    for row in table:
        writer.writerow(row)
  

if __name__=="__main__":
    main()
