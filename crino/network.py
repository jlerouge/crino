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

"""
provides some ready-to-use neural networks, such as MLP's, DNN's and IODA's. 
"""

import datetime as DT
import theano
import theano.tensor as T
import numpy as np
import sys

from crino.module import Sequential, Linear, Sigmoid, Tanh
from crino.criterion import CrossEntropy, MeanSquareError


class MultiLayerPerceptron(Sequential):
    """
    A `MultiLayerPerceptron` (MLP) is one classical form of artificial
    neural networks, whichs aims at predicting one or more output states
    given some particular inputs. A MLP is a `Sequential` module, made of
    a succession of `Linear` modules and non-linear `Activation` modules.
    This tends to make the MLP able to learn non-linear decision functions.

    A MLP must be trained with a supervised learning algorithm in order
    to work. The gradient backpropagation is by far the most used algorithm
    used to train MLPs.
    """
    def __init__(self, nUnits, outputActivation=Sigmoid):
        """
        Constructs a new `MultiLayerPerceptron` network.

        :Parameters:
            nUnits : int list
                The sizes of the (input, hidden and output) representations.
            outputActivation : class derived from `Activation`
                The type of activation for the output layer.
        :attention: `outputActivation` parameter is not an instance but a class.
        """
        Sequential.__init__(self, nInputs=nUnits[0])
        self.nUnits = nUnits

        # In and hidden layers
        for nOutputs in nUnits[1:-1]:
            self.add(Linear(nOutputs))
            self.add(Tanh(nOutputs))
        # Output layer
        self.add(Linear(nUnits[-1]))
        self.add(outputActivation(nUnits[-1]))
        

    def getGeometry(self):
        if not(self.prepared):
            raise  ValueError("You can not get geometry on a non-prepared MLP")
        geometry=[self.nInputs]
        geometry+=list(map(lambda mod:mod.nOutputs,self.modules))
        
        return geometry

    def getParameters(self):
        if not(self.prepared):
            raise  ValueError("You can not get params on a non-prepared MLP")
        params={}
        params['geometry']=self.getGeometry()
        params['weights']=list(map(lambda param:np.array(param.get_value()),self.params))
  
        return params 

    def setParameters(self,params):
        if not(self.prepared):
            raise  ValueError("You can not set params on a non-prepared MLP")
        if self.getGeometry()!=params['geometry']:
            raise  ValueError("Params geometry does not match MLP geoemtry")
        
        for param,w in zip(self.params,params['weights']):
            param.set_value(w)
        
    def initEpochHook(self):
        pass
        
    def checkEpochHook(self):
        #print(self.finetunevars)
        return False

    def initBadmoveHook(self):
        pass
        
    def checkBadmoveHook(self):
        #print(self.finetunevars)
        return False

    def initBatchHook(self):
        pass
        
    def checkBatchHook(self):
        #print(self.finetunevars)
        return False
    
    def checkLearningParameters(self,param_dict):
        known={}
        unknown={}
        known_keys=["batch_size","learning_rate", "epochs", "growth_factor", "growth_threshold","badmove_threshold","verbose"]
        for (key, value) in param_dict.items():
            if key in known_keys:
                known[key]=value
            else:
                unknown[key]=value
        return (known,unknown)
    
    def defaultLearningParameters(self,param_dict):
        ret=dict(param_dict)
        default_values={'batch_size':1, 'learning_rate':1.0, 'epochs':100, 'growth_factor':1.25, 'growth_threshold':5, 'badmove_threshold':10, 'verbose':True}
        for key in default_values.keys():
            if not(param_dict.has_key(key)):
                ret[key]=default_values[key]
        return ret

    def finetune(self, shared_x_train, shared_y_train, batch_size, learning_rate, epochs, growth_factor, growth_threshold, badmove_threshold, verbose):
        """
        Performs the supervised learning step of the `MultiLayerPerceptron`,
        using a batch-gradient backpropagation algorithm. The `learning_rate`
        is made adaptative with the `growth_factor` multiplier. If the mean loss
        a number is improved during `growth_threshold` successive epochs, then
        the `learning_rate` is increased, and if the mean loss is degraded during
        one epoch, then the `learning_rate` is decreased.

        :Parameters:
            x_train : SharedVariable from :numpy:`ndarray`
                The training examples.
            y_train : SharedVariable from :numpy:`ndarray`
                The training labels.
            batch_size : int
                The number of training examples in each mini-batch.
            learning_rate : float
                The rate used to update the parameters with the gradient.
            epochs : int
                The number of epochs to run the training algorithm.
            growth_factor : float
                The multiplier factor used to increase or decrease the `learning_rate`.
            growth_threshold : int
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            badmove_threshold : int
                The number of successive loss-non-improving gradient descents after which parameters must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.

        :return: elapsed time, in datetime.
        """
        
        self.finetunevars={}
        lvar=self.finetunevars
        
        lvar['shared_x_train']=shared_x_train
        lvar['shared_y_train']=shared_y_train
        lvar['batch_size']=batch_size
        lvar['learning_rate']=learning_rate
        lvar['epochs']=epochs
        lvar['growth_factor']=growth_factor
        lvar['growth_threshold']=growth_threshold 
        lvar['badmove_threshold']=badmove_threshold
        lvar['verbose']=verbose
        
        # Compilation d'une fonction theano pour l'apprentissage du modèle
        train = self.trainFunction(lvar['batch_size'], lvar['learning_rate'], True, lvar['shared_x_train'], lvar['shared_y_train'])
        hold=self.holdFunction()
        restore=self.restoreFunction()
        lvar['n_train_batches'] = lvar['shared_x_train'].get_value().shape[0]/lvar['batch_size']
        lvar['start_time'] = DT.datetime.now()
        lvar['mean_loss'] = float('inf')
        lvar['good_epochs'] = 0
        self.finetune_history=[(-1,lvar['learning_rate'],lvar['mean_loss'])]
        
        self.initEpochHook()
        for lvar['epoch'] in xrange(lvar['epochs']):
            lvar['epoch_start']=DT.datetime.now()
            lvar['loss_by_batch'] = []
            hold()
            if(lvar['verbose']):
                print "",
            
            self.initBadmoveHook()            
            for lvar['badmoves'] in xrange(lvar['badmove_threshold']):
                
                self.initBatchHook()
                for lvar['batch_index'] in xrange(lvar['n_train_batches']):
                    loss = train(lvar['batch_index'])
                    lvar['loss_by_batch'].append(loss)
                    if(lvar['verbose']):
                        print "\r  | |_Batch %d/%d, loss : %f" % (lvar['batch_index']+1, lvar['n_train_batches'], loss),
                        sys.stdout.flush()
                    if self.checkBatchHook():
                        break                    

                lvar['new_mean_loss'] = np.mean(lvar['loss_by_batch'])
                self.finetune_history.append((lvar['epoch'],lvar['learning_rate'],lvar['new_mean_loss']))
                
                if self.checkBadmoveHook():
                    break 
                
                if  lvar['new_mean_loss'] < lvar['mean_loss']:
                    lvar['good_epochs'] += 1
                    break                
                
                if lvar['badmoves']+1<lvar['badmove_threshold']:
                    if(lvar['verbose']):
                        print "\r# Bad move %f > %f; Learning rate : %f > %f" % (lvar['mean_loss'], lvar['new_mean_loss'], lvar['learning_rate'], lvar['learning_rate']/lvar['growth_factor'])                    
                    restore()                    
                    lvar['learning_rate'] = lvar['learning_rate']/lvar['growth_factor']
                    train = self.trainFunction(
                        lvar['batch_size'], lvar['learning_rate'],True,
                        lvar['shared_x_train'], lvar['shared_y_train'])
                else:
                    if(lvar['verbose']):
                        print("\r# Break Epoch on bad move threshold")
                        
                lvar['good_epochs'] = 0


            lvar['mean_loss'] = lvar['new_mean_loss']

            if(lvar['good_epochs'] >= lvar['growth_threshold']):
                lvar['good_epochs'] = 0
                if(lvar['verbose']):
                    print "\r# Fast Track; Learning rate : %f > %f" % (lvar['learning_rate'], lvar['learning_rate']*lvar['growth_factor'])
                lvar['learning_rate'] = lvar['learning_rate']*lvar['growth_factor']
                train = self.trainFunction(
                    lvar['batch_size'], lvar['learning_rate'], True,
                    lvar['shared_x_train'], lvar['shared_y_train'])
           
            if(lvar['verbose']):
                print "\r  |_Epoch %d/%d, mean loss : %f, duration (s) : %s" % (lvar['epoch']+1, lvar['epochs'], lvar['new_mean_loss'],(DT.datetime.now()-lvar['epoch_start']).total_seconds())
            
            if self.checkEpochHook():
                break
                
        return (DT.datetime.now()-lvar['start_time'])

    def train(self, x_train, y_train, **params):
        """
        Performs the supervised learning step of the `MultiLayerPerceptron`.
        This function explicitly calls `finetune`, but displays a bit more information.

        :Parameters:
            x_train : :numpy:`ndarray`
                The training examples.
            y_train : :numpy:`ndarray`
                The training labels.
            batch_size : int
                The number of training examples in each mini-batch.
            learning_rate : float
                The rate used to update the parameters with the gradient.
            epochs : int
                The number of epochs to run the training algorithm.
            growth_factor : float
                The multiplier factor used to increase or decrease the `learning_rate`.
            growth_threshold : int
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            badmove_threshold : int
                The number of successive loss-non-improving gradient descents after which parameters must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.

        :return: elapsed time, in datetime.
        :see: `finetune`
        """
        (learning_params,unknown)=self.checkLearningParameters(params)
        if len(unknown)>0:
            print("Waring unknown training parameters %s"%(unknown,))
            
        learning_params=self.defaultLearningParameters(learning_params)

        print "-- Beginning of fine-tuning (%d epochs) --" % (learning_params['epochs'])
        shared_x_train=theano.shared(x_train)
        shared_y_train=theano.shared(y_train)        
        delta = self.finetune(shared_x_train, shared_y_train, **learning_params)
        print "-- End of fine-tuning (lasted %s) --" % (delta)
        return delta


class AutoEncoder(MultiLayerPerceptron):
    """
    An `AutoEncoder` is a neural network whichs aims at encoding
    its inputs in a smaller representation space. It is made of
    a projection layer and a backprojection layer. The compressed
    representation (or hidden representation) lies in the projection
    layer, while the backprojection layer reconstructs the original inputs.

    The weights between those two layers are shared, that means
    that the backprojection matrix is constrained to be the transpose
    of the projection matrix. However, the two biases are independant.

    If the data allows it, the `AutoEncoder` is best learned with a `Sigmoid`
    final activation module in conjunction with a `CrossEntropy` criterion.

    :see: `CrossEntropy`, `Sigmoid`
    """
    def __init__(self, nVisibles, nHidden, outputActivation=Sigmoid):
        """
        Constructs a new `AutoEncoder` network.

        :Parameters:
            nVisibles : int
                The `visibles` size.
            nHidden : int
                The size of the hidden representation.
            outputActivation : class derived from `Activation`
                The type of activation for the backprojection layer.
        :attention: `outputActivation` parameter is not an instance but a class.
        """
        Sequential.__init__(self, nInputs=nVisibles)
        self.nHidden = nHidden
        self.add(Linear(nHidden, nVisibles))
        self.add(Tanh(nHidden))
        self.add(Linear(nVisibles, nHidden))
        self.add(outputActivation(nVisibles))

    def prepareParams(self):
        if(self.modules):
            self.modules[0].prepareParams()
            self.modules[2].prepareParams(self.modules[0].params[0].T)
            self.params.extend(self.modules[0].params)
            self.params.extend([self.modules[2].params[1]])

    def hiddenValues(self, x_input):
        """
        Returns the hidden representation for a given input.

        :Parameters:
            x_input : :numpy:`ndarray`
                The input on which the hidden representation must be computed.

        :return: the corresponding hidden representation
        """
        linear=self.modules[0].forward(x_input)
        return self.modules[1].forward(linear)
    
class OutputAutoEncoder(AutoEncoder):

    def prepareParams(self):
        if(self.modules):
            self.modules[2].prepareParams()
            self.modules[0].prepareParams(self.modules[2].params[0].T)
            self.params.extend(self.modules[2].params)
            self.params.extend([self.modules[0].params[1]])        

class PretrainedMLP(MultiLayerPerceptron):
    """
    A `PretrainedMLP` is a specialization of the MLP, where the
    layers are pretrained, for input part on the training examples
    (:math:`\mathbf{x}) or for output part on the training labels   
    (:math:`\mathbf{y}) using a Stacked `AutoEncoder` strategy.

    :see: `MultiLayerPerceptron`, http://www.deeplearning.net/tutorial/SdA.html
    """
    def __init__(self, inputRepresentationSize, outputRepresentationSize, outputActivation=Sigmoid, nUnitsInput=[], nUnitsLink=[], nUnitsOutput=[]):
        """
        Constructs a new `DeepNeuralNetwork`.

        :Parameters:
            inputRepresentationSize: int
                The size of the input representation
            outputRepresentationSize: int
                The size of the output representation                
            nUnitsInput : int list
                The hidden representations size list on the input side that will be pretrained.
            nUnitsOutput : int list
                The hidden representations size list on the output side that will be pretrained.
            nUnitsLink : int list
                The size list of hidden representations  in-between input and output representations.
            outputActivation : class derived from `Activation`
                The type of activation for the output layer.
        :attention: `outputActivation` parameter is not an instance but a class.
        """
        MultiLayerPerceptron.__init__(self, [inputRepresentationSize]+ nUnitsInput + nUnitsLink + nUnitsOutput + [outputRepresentationSize], outputActivation)
        
        self.inputRepresentationSize=inputRepresentationSize
        self.outputRepresentationSize=outputRepresentationSize
        self.nUnitsInput = nUnitsInput
        self.nUnitsLink= nUnitsLink
        self.nUnitsOutput = nUnitsOutput

        # Construction des AutoEncoders
        self.inputAutoEncoders = []
        self.outputAutoEncoders = []
        self.linkLayer=[]

        x = T.matrix('x')
        y = T.matrix('y')
        
        if len(self.nUnitsInput)>0:
            for nVisibles,nHidden in zip([self.inputRepresentationSize]+self.nUnitsInput[:-1], self.nUnitsInput):
                ae = AutoEncoder(nVisibles, nHidden)
                ae.linkInputs(x,nVisibles)
                ae.prepare()
                ae.criterion = CrossEntropy(ae.outputs, y)
                self.inputAutoEncoders.append(ae)
            self.lastInputSize=self.nUnitsInput[-1]
        else:
            self.lastInputSize=self.inputRepresentationSize

        if len(self.nUnitsOutput)>0:
            for nHidden,nVisibles in zip(self.nUnitsOutput, self.nUnitsOutput[1:]+[self.outputRepresentationSize]):
                ae = OutputAutoEncoder(nVisibles, nHidden)
                ae.linkInputs(x,nVisibles)
                ae.prepare()
                ae.criterion = CrossEntropy(ae.outputs, y)
                self.outputAutoEncoders.append(ae)
            self.firstOutputSize=self.nUnitsOutput[0]
            linkLayerLastActivation=Tanh
        else:
            self.firstOutputSize=self.outputRepresentationSize                
            linkLayerLastActivation=outputActivation
   
        self.linkLayer=MultiLayerPerceptron([self.lastInputSize]+nUnitsLink+[self.firstOutputSize], linkLayerLastActivation)
        self.linkLayer.linkInputs(x,self.lastInputSize)
        self.linkLayer.prepare()
        
            
        self.linkInputData=None
        self.linkOutputData=None

    def prepareParams(self):
        if(self.modules):
            if len(self.nUnitsInput)>0:
                inputParams=list(map(lambda ae:(ae.params[0], ae.params[1]),self.inputAutoEncoders))
            else:
                inputParams=[]
            
            linkParams=list(map(lambda mod:(mod.params[0], mod.params[1]),self.linkLayer.modules[::2]))
            
            if len(self.nUnitsOutput)>0:
                outputParams=list(map(lambda ae:(ae.params[0], ae.params[1]),self.outputAutoEncoders))
            else:
                outputParams=[]            
                
            for mod,params in zip(self.modules[::2],inputParams+linkParams+outputParams):
                mod.prepareParams(params[0],params[1])
                self.params.extend([mod.params[0],mod.params[1]])
                

    def pretrainInputAutoEncoders(self, data, **params):
        """
        Performs the unsupervised learning step of the  input autoencoders,
        using a batch-gradient backpropagation algorithm.
        
        Classically, in a `DeepNeuralNetwork`, only the input autoencoders are pretrained.
        The data used for this pretraining step can be the input training dataset
        used for the supervised learning (see `finetune`), or a subset of this dataset,
        or else a specially crafted input pretraining dataset.        

        Once an `AutoEncoder` is learned, the projection (encoding) layer is kept and used
        to initialize the network layers. The backprojection (decoding) part is not useful
        anymore.

        The `learning_rate` is made adaptative with the `growth_factor` multiplier.
        If the mean loss a number is improved during `growth_threshold` successive epochs,
        then the `learning_rate` is increased, and if the mean loss is degraded during
        one epoch, then the `learning_rate` is decreased.

        :Parameters:
            data : SharedVariable from :numpy:`ndarray`
                The training data (typically example features).
            batch_size : int
                The number of training examples in each mini-batch.
            learning_rate : float
                The rate used to update the parameters with the gradient.
            epochs : int
                The number of epochs to run the training algorithm.
            growth_factor : float
                The multiplier factor used to increase or decrease the `learning_rate`.
            growth_threshold : int
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            badmove_threshold : int
                The number of successive loss-non-improving gradient descents after which parameters must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.
        """
        (learning_params,unknown)=self.checkLearningParameters(params)
        if len(unknown)>0:
            print("Waring unknown training parameters %s"%(unknown,))
            
        learning_params=self.defaultLearningParameters(learning_params)
            
        n_train_batches = data.shape[0]/learning_params['batch_size']
        shared_inputs= data
        start_time = DT.datetime.now()
        for (ae,layer) in zip(self.inputAutoEncoders, xrange(len(self.inputAutoEncoders))):
            if learning_params['verbose']:
                print("--- Learning input AE %d/%d ---"%(layer+1,len(self.inputAutoEncoders)))
            delta=ae.finetune(shared_inputs, shared_inputs, **learning_params)
            if learning_params['verbose']:
                print("--- Input AE %d/%d Learned, Duration (s) %d ---"%(layer+1,len(self.inputAutoEncoders),delta.total_seconds()))            
            inputs = (ae.hiddenValues(shared_inputs.get_value())+1.)/2.
            shared_inputs=theano.shared(inputs)
        self.linkInputData=shared_inputs
        return (DT.datetime.now()-start_time)
    
    def pretrainOutputAutoEncoders(self, data, **params):
        """
        Performs the unsupervised learning step of the output autoencoders,
        using a batch-gradient backpropagation algorithm.

        The `OutputOutputDeepArchitecture`  pretrains the output autoencoders,
        in the same way the `DeepNeuralNetwork` does for input autoencoders. In
        this case, the given training data are the labels (:math:`\mathbf{y}`)
        and not the examples (:math:`\mathbf{x}`) (i.e. the labels that the network
        must predict).

        Once an `AutoEncoder` is learned, the backprojection layer (decoding) is kept and used
        to initialize the network layers. The projection (encoding) part is not useful
        anymore.

        The `learning_rate` is made adaptative with the `growth_factor` multiplier.
        If the mean loss a number is improved during `growth_threshold` successive epochs,
        then the `learning_rate` is increased, and if the mean loss is degraded during
        one epoch, then the `learning_rate` is decreased.

        :Parameters:
            data : SharedVariable from :numpy:`ndarray`
                The training data (typically example labels).
            batch_size : int
                The number of training examples in each mini-batch.
            learning_rate : float
                The rate used to update the parameters with the gradient.
            epochs : int
                The number of epochs to run the training algorithm.
            growth_factor : float
                The multiplier factor used to increase or decrease the `learning_rate`.
            growth_threshold : int
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            badmove_threshold : int
                The number of successive loss-non-improving gradient descents after which parameters must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.
        """
        (learning_params,unknown)=self.checkLearningParameters(params)
        if len(unknown)>0:
            print("Waring unknown training parameters %s"%(unknown,))
            
        learning_params=self.defaultLearningParameters(learning_params)
            
        n_train_batches = data.shape[0]/learning_params['batch_size']
        shared_outputs= data
        start_time = DT.datetime.now()
        for (ae,layer) in reversed(zip(self.outputAutoEncoders, xrange(len(self.outputAutoEncoders)))):
            if learning_params['verbose']:
                print("--- Learning output AE %d/%d ---"%(layer+1,len(self.outputAutoEncoders)))
            delta=ae.finetune(shared_outputs, shared_outputs, **learning_params)
            if learning_params['verbose']:
                print("--- Output AE %d/%d Learned, Duration (s) %d ---"%(layer+1,len(self.outputAutoEncoders),delta.total_seconds()))            
            outputs = ae.hiddenValues(shared_outputs.get_value())
            shared_outputs=theano.shared((outputs+1.)/2.)
        self.linkOutputData=theano.shared(outputs)
        return (DT.datetime.now()-start_time)   

    def train(self, x_train, y_train, **params):
        """
        Performs the pretraining step for the input and output autoencoders,
        optionally the semi-supervised pretraining step of the link layer, and finally the supervised learning
        step (`finetune`).

        :Parameters:
            x_train : :numpy:`ndarray`
                The training examples.
            y_train : :numpy:`ndarray`
                The training labels.
            batch_size : int
                The number of training examples in each mini-batch.
            learning_rate : float
                The rate used to update the parameters with the gradient.
            epochs : int
                The number of epochs to run the training algorithm.
            growth_factor : float
                The multiplier factor used to increase or decrease the `learning_rate`.
            growth_threshold : int
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            badmove_threshold : int
                The number of successive loss-non-improving gradient descents after which parameters must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.
            input_pretraining_params : dict
                The paramaters the pretraining step,
                possible keys: batch_size, learning_rate, epochs, growth_factor, growth_threshold, badmove_threshold, verbose 
            output_pretraining_params : dict
                The paramaters the pretraining step,
                possible keys: batch_size, learning_rate, epochs, growth_factor, growth_threshold, badmove_threshold, verbose
            link_pretraining : bool
                Should we pretrain the link layer (Default: False)
            link_pretraining_params : dict
                The paramaters the pretraining step,
                possible keys: batch_size, learning_rate, epochs, growth_factor, growth_threshold, badmove_threshold, verbose                

        :return: elapsed time, in deltatime.
        :see: `pretrainAutoEncoders`, `pretrainLink`, `finetune`
        """

        (training_params,unknown)=self.checkLearningParameters(params)
        if unknown.has_key("input_pretraining_params"):
            input_pretraining_params=unknown.pop("input_pretraining_params")        
        else:
            input_pretraining_params={'epochs':50}
        
        if unknown.has_key("output_pretraining_params"):
            output_pretraining_params=unknown.pop("output_pretraining_params")        
        else:
            output_pretraining_params={'epochs':50}

        if unknown.has_key("link_pretraining"):
            link_pretraining=unknown.pop("link_pretraining")        
        else:
            link_pretraining=False
            
        if unknown.has_key("link_pretraining_params"):
            link_pretraining_params=unknown.pop("link_pretraining_params")        
        else:
            link_pretraining_params={'epochs':50}
            
        if len(unknown)>0:
            print("Waring unknown training parameters %s"%(unknown,))

        training_params=self.defaultLearningParameters(training_params)
        input_pretraining_params=self.defaultLearningParameters(input_pretraining_params)
        output_pretraining_params=self.defaultLearningParameters(output_pretraining_params)  
        link_pretraining_params=self.defaultLearningParameters(link_pretraining_params)  

        shared_x_train=theano.shared(x_train)
        shared_y_train=theano.shared(y_train)
        
        totalDelta=DT.timedelta(0)
        if len(self.nUnitsInput)>0:
            if(training_params['verbose']):
                print "-- Beginning of input layers pre-training (%d epochs) --" % (input_pretraining_params['epochs'])
            delta = self.pretrainInputAutoEncoders(shared_x_train,**input_pretraining_params)
            totalDelta += delta
            if(training_params['verbose']):
                print "-- End of input layers pre-training (lasted %s) --" % (totalDelta)

        if len(self.nUnitsOutput)>0:
            if(training_params['verbose']):            
                print "-- Beginning of output layers pre-training (%d epochs) --" % (output_pretraining_params['epochs'])
            delta = self.pretrainOutputAutoEncoders(shared_y_train, **output_pretraining_params)
            totalDelta += delta
            if(training_params['verbose']):
                print "-- End of output layers pre-training (lasted %s) --" % (delta)
            
        if(link_pretraining):
            if self.linkInputData is None:
                self.linkInputData=shared_x_train
            if self.linkOutputData is None:
                self.linkOutputData=shared_y_train
            
            y = T.matrix('y')
            if len(self.nUnitsOutput)>0:
                self.linkLayer.criterion=MeanSquareError(self.linkLayer.outputs, y)
            else:
                self.linkLayer.criterion=self.criterion.__class__(self.linkLayer.outputs, y)
            
            if(training_params['verbose']):
                print "-- Beginning of link layer pre-training (%d epochs) --" % (link_pretraining_params['epochs'])
            delta = self.linkLayer.finetune(self.linkInputData,self.linkOutputData,**link_pretraining_params)
            totalDelta += delta
            if(training_params['verbose']):
                print "-- End of link layer pre-training (lasted %s) --" % (delta)

        if(training_params['verbose']):
            print "-- Beginning of fine-tuning (%d epochs) --" % (training_params['epochs'])
        delta = self.finetune(shared_x_train, shared_y_train, **training_params)
        totalDelta += delta
        if(training_params['verbose']):
            print "-- End of fine-tuning (lasted %s) --" % (delta)
        return totalDelta
    
class DeepNeuralNetwork(PretrainedMLP):
    """
    A `DeepNeuralNetwork` (DNN) is a specialization of the MLP, where the
    layers are pretrained on the training examples (:math:`\mathbf{x})
    using a Stacked `AutoEncoder` strategy. It has been specifically designed
    for data that lies in a high-dimensional input space.

    :see: `MultiLayerPerceptron`, http://www.deeplearning.net/tutorial/SdA.html
    """
    def __init__(self, nUnitsInput, nUnitsOutput, outputActivation=Sigmoid):
        """
        Constructs a new `DeepNeuralNetwork`.

        :Parameters:
            nUnitsInput : int list
                The sizes of the (input and hidden) representations on the input side.
            nUnitsOutput : int list
                The sizes of the (hidden and output) representations on the output side.
            outputActivation : class derived from `Activation`
                The type of activation for the output layer.
        :attention: `outputActivation` parameter is not an instance but a class.
        """    
        PretrainedMLP.__init__(self,
            inputRepresentationSize=nUnitsInput[0], outputRepresentationSize=nUnitsOutput[-1],
            outputActivation=outputActivation, nUnitsInput=nUnitsInput[1:], nUnitsLink=nUnitsOutput[:-1], nUnitsOutput=[])
    
class InputOutputDeepArchitecture(PretrainedMLP):
    """
    An `InputOutputDeepArchitecture` (IODA) is a specialization of the DNN,
    where the layers are divided into three categories : the input layers,
    the link layer and the output layers. It has been specifically designed
    for cases where both the input and the output spaces are high-dimensional.

    The input and output layers are pretrained on the training example
    (:math:`\mathbf{x}) and the training labels (:math:`\mathbf{y}),
    respectively, using a Stacked `AutoEncoder` strategy, as for DNNs.

    The link layer can optionally be pretrained, using as input and output data
    the hidden representations of the deepmost input and output autoencoders,
    respectively.

    :see: `DeepNeuralNetwork`, http://www.deeplearning.net/tutorial/SdA.html
    """

    def __init__(self, nUnitsInput, nUnitsOutput, outputActivation=Sigmoid):
        """
        Constructs a new `InputOutputDeepArchitecture`.

        :Parameters:
            nUnitsInput : int list
                The sizes of the (input and hidden) representations on the input side.
            nUnitsOutput : int list
                The sizes of the (hidden and output) representations on the output side.
            outputActivation : class derived from `Activation`
                The type of activation for the output layer.
        :attention: `outputActivation` parameter is not an instance but a class.
        """
        PretrainedMLP.__init__(self,
            inputRepresentationSize=nUnitsInput[0], outputRepresentationSize=nUnitsOutput[-1],
            outputActivation=outputActivation, nUnitsInput=nUnitsInput[1:], nUnitsLink=[], nUnitsOutput=nUnitsOutput[:-1])   