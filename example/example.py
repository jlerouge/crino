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
import json
import csv

import datetime as DT

def defaultConfig():

    config={}

    #Learning parameters of the input pretraining
    input_pretraining_params={
            'learning_rate': 10.0,
            'batch_size' : 100,
            'epochs' : 300
            }

    #Learning parameters of the output pretraining
    output_pretraining_params={
            'learning_rate': 10.0,
            'batch_size' : 100,
            'epochs' : 300
            }

    ##Learning parameters of the link pretraining
    #link_pretraining_params={
            #'learning_rate': 1.0,
            #'batch_size' : 100,
            #'epochs' : 300
            #}

    #Learning parameters of the supervised training + pretrainings
    config['learning_params']={
        'learning_rate' : 2.0,
        'batch_size' : 100,
        'epochs' : 300,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        #'link_pretraining_params' : link_pretraining_params,
        'link_pretraining' : False
    }

    #Size of one hidden representation
    hidden_size = 1024
    #Geometry of all hidden representations
    config['hidden_geometry'] = [hidden_size,hidden_size]

    #How many layers are pretrained
    # (here 1 at input and 1 at output)
    config['pretraining_geometry']={
        'nInputLayers': 1,
        'nOutputLayers': 1
    }

    #Shall we used known init weights (here no)
    config['init_weights'] = None
    #Shall we save init weights
    config['save_init_weights'] = True

    #Examples to be displayed at testing
    config['displayed_examples']=[10,50,100]

    #Epochs to be displayed at testing
    config['displayed_epochs']=[0,10,100,200,300]

    #Where to store results
    config['outfolder']='./default_config_example_results-%s/'%(DT.datetime.now().strftime("%Y-%m-%d-%H-%M"),)

    return config



class MyPretrainedMLP(PretrainedMLP):
# We subclass the Pretrained MLP class to gther some information during
# learning phase and to folow the evolution of the criterion on an other
# set than the training set
    def setDisplayedEpochs(self,displayed_epochs):
    # set the epochs where we will compute a forward for the seperate set
        self.displayed_epochs=displayed_epochs

    def setTestSet(self,x_test,y_test):
    # set the seperate set not used as training set but used for displaying what is happening
        self.shared_x_test=theano.shared(x_test)
        self.shared_y_test=theano.shared(y_test)

    def initEpochHook(self,finetune_vars):
    # initialize storage variables before starting the great learning loop
        self.testCriterionFunction=self.criterionFunction(downcast=True, shared_x_data=self.shared_x_test, shared_y_data=self.shared_y_test)
        self.testForwardFunction=self.forwardFunction(downcast=True, shared_x_data=self.shared_x_test)

        self.test_criterion_history=[np.mean(self.testCriterionFunction())]
        self.test_forward_history=[(-1,self.testForwardFunction())]

        self.appForwardFunction=self.forwardFunction(downcast=True, shared_x_data=finetune_vars['shared_x_train'])
        self.app_forward_history=[(-1,self.appForwardFunction())]

    def checkEpochHook(self,finetune_vars):
    # compute the criterion (whithout backprop) on the separate set
        self.test_criterion_history.append(np.mean(self.testCriterionFunction()))
        if finetune_vars['epoch']+1 in self.displayed_epochs:
            # compute a forward pass only on certain epochs and sotre the results
            self.test_forward_history.append((finetune_vars['epoch'],self.testForwardFunction()))
            self.app_forward_history.append((finetune_vars['epoch'],self.appForwardFunction()))


class MyValidPretrainedMLP(PretrainedMLP):
# We subclass the Pretrained MLP class to gther some information during
# learning phase and to folow the evolution of the criterion on an other
# set than the training set
    def setDisplayedEpochs(self,displayed_epochs):
    # set the epochs where we will compute a forward for the seperate set
        self.displayed_epochs=displayed_epochs

    def setTestSet(self,x_test,y_test):
    # set a seperate test set
        self.shared_x_test=theano.shared(x_test)
        self.shared_y_test=theano.shared(y_test)

    def setValidSet(self,x_valid,y_valid):
    # set the seperate valid set
        self.shared_x_valid=theano.shared(x_valid)
        self.shared_y_valid=theano.shared(y_valid)

    def setValidThreshold(self,n):
    # set the number of epochs with a valid error above the valid error min to accept without breaking
        self.valid_threshold=n

    def initEpochHook(self,finetune_vars):
    # initialize storage variables before starting the great learning loop

        self.testCriterionFunction=self.criterionFunction(downcast=True, shared_x_data=self.shared_x_test, shared_y_data=self.shared_y_test)
        self.testForwardFunction=self.forwardFunction(downcast=True, shared_x_data=self.shared_x_test)

        self.validCriterionFunction=self.criterionFunction(downcast=True, shared_x_data=self.shared_x_valid, shared_y_data=self.shared_y_valid)
        self.validForwardFunction=self.forwardFunction(downcast=True, shared_x_data=self.shared_x_valid)

        self.valid_criterion_history=[np.mean(self.validCriterionFunction())]
        self.valid_forward_history=[(-1,self.validForwardFunction())]
        self.valid_error_min=None

        self.appForwardFunction=self.forwardFunction(downcast=True, shared_x_data=finetune_vars['shared_x_train'])
        self.app_forward_history=[(-1,self.appForwardFunction())]

        self.break_on_epoch=None


    def checkEpochHook(self,finetune_vars):
    # compute the criterion (whithout backprop) on the separate set
        valid_error=np.mean(self.validCriterionFunction())
        self.valid_criterion_history.append(valid_error)
        if (self.valid_error_min is None ) or (self.valid_error_min > valid_error) :
            self.valid_error_min=valid_error

        if finetune_vars['epoch']+1 in self.displayed_epochs:
            # compute a forward pass only on certain epochs and sotre the results
            self.valid_forward_history.append((finetune_vars['epoch'],self.validForwardFunction()))
            self.app_forward_history.append((finetune_vars['epoch'],self.appForwardFunction()))

        if (np.array(self.valid_criterion_history[-self.valid_threshold:]) > self.valid_error_min).all():
            self.break_on_epoch=finetune_vars['epoch']
            return True
        else:
            return False


def data2greyimg(filename, X):
# Convinient functoin to save 2D array as png
    Xn=(X-X.min())/(X.max()-X.min())*255
    scipy.misc.imsave(filename, Xn)


def experience(config):

    needed_params=['learning_params','hidden_geometry','pretraining_geometry','init_weights','save_init_weights','displayed_examples','displayed_epochs','outfolder']

    used_config={}
    for aParam in needed_params:
        if not( aParam in config.keys()):
            raise ValueError("Experience configuration does not contain %s parameter"%(aParam))
        if aParam!='init_weights':
            used_config[aParam]=config[aParam]
        elif not (config[aParam] is None):
            used_config['init_weights']= []

    learning_params=config['learning_params']
    hidden_geometry=config['hidden_geometry']
    pretraining_geometry=config['pretraining_geometry']
    init_weights=config['init_weights']
    save_init_weights=config['save_init_weights']
    displayed_examples=config['displayed_examples']
    displayed_epochs=config['displayed_epochs']
    outfolder=config['outfolder']

    absoutfolder=os.path.abspath(outfolder)
    if not os.path.exists(absoutfolder):
        os.mkdir(absoutfolder)

    mystdoutpath=os.path.join(absoutfolder,"experience.log")
    print('switch stdout to %s'%(mystdoutpath,))
    mystdout=open(mystdoutpath,'wb')
    sys.stdout=mystdout

    print('... saving used configuration')
    json.dump(used_config,open(os.path.join(absoutfolder,"configuration.json"),'wb'),indent=2)

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

    # Input representation size is the number of pixel
    nInputs=nFeats
    # Output representation size is the number of pixel
    nOutputs=nFeats

    # Compute the full geometry of the MLP
    geometry=[nFeats] + hidden_geometry+[nFeats]
    # Compute the number of layers
    nLayers=len(geometry)-1


    for phase,xdata,ydata in [['train',x_train,y_train],['test',x_test,y_test]]:
        for ex in displayed_examples:
            x_orig = np.reshape(xdata[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_input.png"%(phase,ex,)),x_orig)
            y_true = np.reshape(ydata[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_target.png"%(phase,ex,)),y_true)


    print '... building and learning a network'
    nn = MyPretrainedMLP(geometry, outputActivation=crino.module.Sigmoid,**pretraining_geometry)

    # set the test set
    nn.setTestSet(x_test,y_test)
    # set the epochs where we will have a particular look at
    nn.setDisplayedEpochs(displayed_epochs)

    # bake the MLP and set the criterion
    nn.setInputs(T.matrix('x'), nFeats)
    nn.prepare()
    nn.criterion = MeanSquareError(nn.outputs, T.matrix('y'))

    # set initial weights if they exists
    if not(init_weights is None):
        nn.setParameters(init_weights)
    # save initial weights if ask
    if save_init_weights:
        pickle.dump(nn.getParameters(),open(os.path.join(absoutfolder,"starting_params.pck"),'w'),protocol=-1)

    delta = nn.train(x_train, y_train, **learning_params)
    print '... learning lasted %s (s) ' % (delta)

    print '... saving results'

    # Save parameters in pythonic serialization
    pickle.dump(nn.getParameters(),open(os.path.join(absoutfolder,"learned_params.pck"),'w'),protocol=-1)

    # Save some history of the learning phase in pythonic serialization
    results={
        'I':pretraining_geometry['nInputLayers'],
        'L':nLayers-pretraining_geometry['nInputLayers']-pretraining_geometry['nOutputLayers'],
        'O':pretraining_geometry['nOutputLayers'],
        'train_criterion':nn.finetune_history[-1],
        'train_history':nn.finetune_history,
        'train_full_history':nn.finetune_full_history,
        'test_criterion': nn.test_criterion_history[-1],
        'test_history':nn.test_criterion_history,
        }
    pickle.dump(results,open(os.path.join(absoutfolder,'results.pck'),'w'),protocol=-1)

    # Save images of displayed_examples at displayed_epochs
    for phase,xdata,ydata,history in [
                ['train',x_train,y_train,nn.app_forward_history],
                ['test',x_test,y_test,nn.test_forward_history]]:
        for ex in displayed_examples:
            for epoch,forward in history:
                y_estim = np.reshape(forward[ex:ex+1], (xSize, xSize), 'F')
                data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_estim_%03d.png"%(phase,ex,epoch+1)),y_estim)


    # Save the sum-up of the experimentation in a csv file
    table=[["Input Pretrained Layers","Link Layers","Output Pretrained Layers", "Epoch","Train", "Test"]]
    for epoch in displayed_epochs:
        table.append([results['I'],results['L'],results['O'],epoch,results['train_history'][epoch],results['test_history'][epoch]])

    writer=csv.writer(open(os.path.join(absoutfolder,'results.csv'),'wb'),delimiter='\t')
    for row in table:
        writer.writerow(row)

    sys.stdout=sys.__stdout__
    print('reverts stdout to console')

def main():
    experience(defaultConfig())

if __name__=="__main__":
    main()
