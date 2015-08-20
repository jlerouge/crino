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

from example_multi_pretraining_strategies import *

import datetime as DT

def fourLayersConfig():

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
    hidden_size = 512
    #Geometry of all hidden representations
    config['hidden_geometry'] = [hidden_size, hidden_size/2, hidden_size]

    # All config['pretraining_geometries'] have the same geometry 3 layers and 4 representations
    # with sizes [nFeats,hidden_size,hidden_size,nFeats].
    # They only differ in the way layers are pretrained or not.
    config['pretraining_geometries']=[]
    # Standard MLP, no pretraining
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':0})
    # First layer pretrained input way
    config['pretraining_geometries'].append({'nInputLayers':1,'nOutputLayers':0})
    # Two first layers pretrained input way
    config['pretraining_geometries'].append({'nInputLayers':2,'nOutputLayers':0})
    # Tree first layers pretrained input way
    config['pretraining_geometries'].append({'nInputLayers':3,'nOutputLayers':0})
    # Last layer pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':1})
    # Two last layers pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':2})
    # Tree last layers pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':0,'nOutputLayers':3})
    # First layer pretrained input way, and last layer pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':1,'nOutputLayers':1})
    # Two First layers pretrained input way, and last layer pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':2,'nOutputLayers':1})
    # First layer pretrained input way, and two last layer2 pretrained output way
    config['pretraining_geometries'].append({'nInputLayers':1,'nOutputLayers':2})

    #Shall we used known init weights (here no)
    config['init_weights'] = None
    #Shall we save init weights
    config['save_init_weights'] = True

    #Examples to be displayed at testing
    config['displayed_examples']=[10,50,100]

    #Epochs to be displayed at testing
    config['displayed_epochs']=[0,10,100,200,300]

    #Where to store results
    config['outfolder']='./results-%d-layers-%d-units-%s/'%(len(config['hidden_geometry'])+1,hidden_size,DT.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    return config



def main():
    experience_multiple_pretraining_geometry(fourLayersConfig())

if __name__=="__main__":
    main()
