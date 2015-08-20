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

import csv

import json

from example import data2greyimg,MyValidPretrainedMLP

import datetime as DT


outformat='pdf'
if outformat=='pdf':
  outback='pdf'
else:
  outback='cairo'
import matplotlib
matplotlib.use(outback)
import matplotlib.pyplot as plt


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
        'epochs' : 2000,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        #'link_pretraining_params' : link_pretraining_params,
        'link_pretraining' : False
    }

    #Size of one hidden representation
    hidden_size = 1024
    #Geometry of all hidden representations
    config['hidden_geometry'] = [hidden_size]*2

    # All configurations have the same geometry 3 layers and 4 representations
    # with sizes [nFeats,hidden_size,hidden_size,nFeats].
    # They only differ in the way layers are pretrained or not.
    config['pretraining_geometries']=[]
    # Standard MLP, no pretraining
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':0})
    # First layer pretrained input way
    config['pretraining_geometries'].append({'nInputLayers':1,'nOutputLayers':0})
    # Two first layers pretrained input way
    config['pretraining_geometries'].append({'nInputLayers':2,'nOutputLayers':0})
    # Last layer pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':1})
    # Two last layers pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':2})
    # First layer pretrained input way, and last layer pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':1,'nOutputLayers':1})

    #Shall we used known init weights (here no)
    config['init_weights'] = None
    #Shall we save init weights
    config['save_init_weights'] = True

    #Examples to be displayed at testing
    config['displayed_examples']=[10,50,100]

    #Epochs to be displayed at testing
    config['displayed_epochs']=[0,10,100,200,300,400,600,700,800,900,1000]

    # Number of epochs before breaking the learning
    config['valid_threshold']=5

    #Where to store results
    config['outfolder']='./results-valid-%d-layers-%d-units-%s/'%(len(config['hidden_geometry'])+1,hidden_size,DT.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    return config


def experience_multiple_pretraining_geometry_and_valid(config):

    needed_params=['learning_params','hidden_geometry','pretraining_geometries','init_weights','save_init_weights','displayed_examples','displayed_epochs','outfolder','valid_threshold']

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
    pretraining_geometries=config['pretraining_geometries']
    init_weights=config['init_weights']
    save_init_weights=config['save_init_weights']
    displayed_examples=config['displayed_examples']
    displayed_epochs=config['displayed_epochs']
    outfolder=config['outfolder']
    valid_threshold=config['valid_threshold']

    absoutfolder=os.path.abspath(outfolder)
    if not os.path.exists(absoutfolder):
        os.mkdir(absoutfolder)

    mystdoutpath=os.path.join(absoutfolder,"experience.log")
    print('switch stdout to %s'%(mystdoutpath,))
    mystdout=open(mystdoutpath,'wb')
    sys.stdout=mystdout

    print '... loading training data'
    train_set = sio.loadmat('data/fixed/train.mat')
    x_train = np.asarray(train_set['x_train'], dtype=theano.config.floatX) # We convert to float32 to
    y_train = np.asarray(train_set['y_train'], dtype=theano.config.floatX) # compute on GPUs with CUDA

    print '... loading valid data'
    valid_set = sio.loadmat('data/fixed/valid.mat')
    x_valid = np.asarray(valid_set['x_valid'], dtype=theano.config.floatX) # We convert to float32 to
    y_valid = np.asarray(valid_set['y_valid'], dtype=theano.config.floatX) # compute on GPUs with CUDA

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

    geometry=[nFeats]+config['hidden_geometry']+[nFeats]
    nLayers=len(geometry)-1

    parameters=init_weights

    if not (parameters is None) and save_init_weights:
        pickle.dump(parameters,open(os.path.join(absoutfolder,"starting_params.pck"),'w'),protocol=-1)

    print('... saving used configuration')
    json.dump(used_config,open(os.path.join(absoutfolder,"configuration.json"),'wb'),indent=2)

    results={}

    for phase,xdata,ydata in [['train',x_train,y_train],['test',x_test,y_test]]:
        for ex in displayed_examples:
            x_orig = np.reshape(xdata[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_input.png"%(phase,ex,)),x_orig)
            y_true = np.reshape(ydata[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_ex_%03d_target.png"%(phase,ex,)),y_true)

    for pretraining_geometry in pretraining_geometries:

        expname="I_%d_L_%d_O_%d"%(pretraining_geometry['nInputLayers'],nLayers-pretraining_geometry['nInputLayers']-pretraining_geometry['nOutputLayers'],pretraining_geometry['nOutputLayers'])

        current_learning_params=dict(learning_params)
        if (learning_params['link_pretraining']) and (pretraining_geometry['nInputLayers']!=0) and (pretraining_geometry['nOutputLayers']!=0):
            current_learning_params['link_pretraining'] = True
        else:
            current_learning_params['link_pretraining'] = False

        print '... building and learning a network %s'%(expname,)
        nn = MyValidPretrainedMLP(geometry, outputActivation=crino.module.Sigmoid,**pretraining_geometry)
        nn.setValidSet(x_valid,y_valid)
        nn.setTestSet(x_test,y_test)
        nn.setValidThreshold(valid_threshold)

        # set the epochs where we will have a particular look at
        nn.setDisplayedEpochs(displayed_epochs)

        nn.linkInputs(T.matrix('x'), nFeats)
        nn.prepare()
        nn.criterion = MeanSquareError(nn.outputs, T.matrix('y'))
        if parameters is None:
            parameters=nn.getParameters()
            if save_init_weights:
                pickle.dump(parameters,open(os.path.join(absoutfolder,"starting_params.pck"),'w'),protocol=-1)
        else:
            nn.setParameters(parameters)
        delta = nn.train(x_train, y_train, **current_learning_params)
        print '... learning lasted %s (s) ' % (delta)

        print('... performing test criterion')
        test_criterion=nn.testCriterionFunction()



        results[expname]={
            'I':pretraining_geometry['nInputLayers'],
            'L':nLayers-pretraining_geometry['nInputLayers']-pretraining_geometry['nOutputLayers'],
            'O':pretraining_geometry['nOutputLayers'],
            'train_criterion':nn.finetune_history[-1],
            'train_history':nn.finetune_history,
            'train_full_history':nn.finetune_full_history,
            'valid_criterion': nn.valid_criterion_history[-1],
            'valid_history':nn.valid_criterion_history,
            'test_criterion': test_criterion,
            'last_epoch': nn.finetune_full_history[-1][0],
            'first_hidden_representation': hidden_geometry[0]
            }
        pickle.dump(nn.getParameters(),open(os.path.join(absoutfolder,"%s_params.pck"%(expname,)),'w'),protocol=-1)

        #print(results[expname])
        print("RESULT %s: train: %f valid: %f test: %f learning epochs %d"%(expname,results[expname]['train_criterion'],results[expname]['valid_criterion'],results[expname]['test_criterion'],results[expname]['last_epoch']+1))

        print("... saving figures")

        for phase,xdata,ydata,history in [
                    ['train',x_train,y_train,nn.app_forward_history],
                    ['valid',x_valid,y_valid,nn.valid_forward_history]]:
            for ex in displayed_examples:
                for epoch,forward in history:
                    y_estim = np.reshape(forward[ex:ex+1], (xSize, xSize), 'F')
                    data2greyimg(os.path.join(absoutfolder,"%s_%s_ex_%03d_estim_%03d.png"%(expname,phase,ex,epoch+1)),y_estim)

        test_forward=nn.testForwardFunction()
        for ex in displayed_examples:
            y_estim = np.reshape(test_forward[ex:ex+1], (xSize, xSize), 'F')
            data2greyimg(os.path.join(absoutfolder,"%s_%s_ex_%03d_estim_%03d.png"%(expname,'test',ex,results[expname]['last_epoch']+1)),y_estim)

        plt.close('all')
        plt.figure(1)
        plt.plot(nn.valid_criterion_history)
        plt.ylabel('criterion')
        plt.xlabel('epochs')
        plt.savefig(os.path.join(absoutfolder,"%s_valid_criterion.pdf"%(expname,)))
        plt.close(1)

    print("... saving results")
    pickle.dump(results,open(os.path.join(absoutfolder,'results.pck'),'w'),protocol=-1)

    table=[["Units","Input Pretrained Layers","Link Layers","Output Pretrained Layers", "Epoch","Train", "Valid"]]
    for expname in results.keys():
        displayed_epochs_before_last=filter(lambda e:e<=results[expname]['last_epoch'],displayed_epochs)
        for epoch in displayed_epochs_before_last:
            table.append([results[expname]['first_hidden_representation'],results[expname]['I'],results[expname]['L'],results[expname]['O'],epoch,results[expname]['train_history'][epoch],results[expname]['valid_history'][epoch]])
        table.append([results[expname]['first_hidden_representation'],results[expname]['I'],results[expname]['L'],results[expname]['O'],
                      results[expname]['last_epoch'],
                      results[expname]['train_criterion'],results[expname]['valid_criterion'],results[expname]['test_criterion']])

    writer=csv.writer(open(os.path.join(absoutfolder,'intremediateresults.csv'),'wb'),delimiter='\t')
    for row in table:
        writer.writerow(row)


    table=[["Units","Input Pretrained Layers","Link Layers","Output Pretrained Layers", "Epoch","Train", "Valid","Test"]]
    for expname in results.keys():
        table.append([results[expname]['first_hidden_representation'],results[expname]['I'],results[expname]['L'],results[expname]['O'],
                      results[expname]['last_epoch'],
                      results[expname]['train_criterion'],results[expname]['valid_criterion'],results[expname]['test_criterion']])

    writer=csv.writer(open(os.path.join(absoutfolder,'finalresults.csv'),'wb'),delimiter='\t')
    for row in table:
        writer.writerow(row)

    sys.stdout=sys.__stdout__
    print('reverts stdout to console')

def main():
    experience_multiple_pretraining_geometry_and_valid(defaultConfig())

if __name__=="__main__":
    main()
