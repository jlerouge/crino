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


from example import *

def fastTestConfig():

    config={}
    
    #Learning parameters of the input pretraining
    input_pretraining_params={
            'learning_rate': 1.0,
            'batch_size' : 250,
            'epochs' : 3
            }
    
    #Learning parameters of the output pretraining
    output_pretraining_params={
            'learning_rate': 1.0,
            'batch_size' : 250,
            'epochs' : 3
            }
    
    #Learning parameters of the link pretraining
    link_pretraining_params={
            'learning_rate': 1.0,
            'batch_size' : 250,
            'epochs' : 3
            }
    
    #Learning parameters of the supervised training + pretrainings
    config['learning_params']={
        'learning_rate' : 1.0,
        'batch_size' : 250,
        'epochs' : 3,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        'link_pretraining_params' : link_pretraining_params,
        'link_pretraining' : True
    }
    
    #Size of one hidden representation
    hidden_size = 256
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
    config['displayed_epochs']=[0,1,3]

    #Where to store results
    config['outfolder']='./fast_test_example_results/'
    
    return config



def main():
    experience(fastTestConfig())

if __name__=="__main__":
    main()