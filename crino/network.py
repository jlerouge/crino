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

"""
provides some ready-to-use neural networks, such as MLP's, DNN's and IODA's. 
"""

import time
import theano
import theano.tensor as T
import numpy as np
import sys

from crino.module import Sequential, Linear, Sigmoid, Tanh
from crino.criterion import CrossEntropy, MeanSquareError

class AutoEncoder(Sequential):
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
    def __init__(self, nInputs, nHidden, outputActivation=Sigmoid):
        """
        Constructs a new `AutoEncoder` network.

        :Parameters:
            nInputs : int
                The `inputs` size.
            nHidden : int
                The size of the hidden representation.
            outputActivation : class derived from `Activation`
                The type of activation for the backprojection layer.
        :attention: `outputActivation` parameter is not an instance but a class.
        """
        Sequential.__init__(self, nInputs=nInputs)
        self.nHidden = nHidden
        self.add(Linear(nHidden, nInputs))
        self.add(Tanh(nHidden))
        self.add(Linear(nInputs, nHidden))
        self.add(outputActivation(nInputs))

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
        return self.modules[1].forward(self.modules[0].forward(x_input))

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

    def finetune(self, x_train, y_train, batch_size=1, learning_rate=1.0, epochs=100, growth_factor=1.25, growth_threshold=5, verbose=True):
        """
        Performs the supervised learning step of the `MultiLayerPerceptron`,
        using a batch-gradient backpropagation algorithm. The `learning_rate`
        is made adaptative with the `growth_factor` multiplier. If the mean loss
        a number is improved during `growth_threshold` successive epochs, then
        the `learning_rate` is increased, and if the mean loss is degraded during
        one epoch, then the `learning_rate` is decreased.

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
            growth_threshold : float
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.

        :return: elapsed time, in seconds.
        """
        # Compilation d'une fonction theano pour l'apprentissage du modèle
        train = self.trainFunction(batch_size, learning_rate, True)
        n_train_batches = x_train.shape[0]/batch_size
        start_time = time.clock()
        mean_loss = float('inf')
        good_epochs = 0
        for epoch in xrange(epochs):
            c = []
            if(verbose):
                print "",
            for i in xrange(n_train_batches):
                loss = train(x_train, y_train, i)
                c.append(loss)
                if(verbose):
                    print "\r  | |_Batch %d/%d, loss : %f" % (i+1, n_train_batches, loss),
                    sys.stdout.flush()

            new_mean_loss = np.mean(c)
            if(new_mean_loss < mean_loss):
                good_epochs += 1
                if(good_epochs >= growth_threshold):
                    good_epochs = 0
                    if(verbose):
                        print "\r# learning rate : %f > %f" % (learning_rate, learning_rate*growth_factor)
                    learning_rate = learning_rate*growth_factor
                    train = self.trainFunction(batch_size, learning_rate, True)
            else:
                good_epochs = 0
                if(verbose):
                    print "\r# learning rate : %f > %f" % (learning_rate, learning_rate/growth_factor)
                learning_rate = learning_rate/growth_factor
                train = self.trainFunction(batch_size, learning_rate, True)
            mean_loss = new_mean_loss
            if(verbose):
                print "\r  |_Epoch %d/%d, mean loss : %f" % (epoch+1, epochs, mean_loss)
        return (time.clock()-start_time)

    def train(self, x_train, y_train, batch_size=1, learning_rate=1.0, epochs=100, growth_factor=1.25, growth_threshold=5, verbose=True):
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
            growth_threshold : float
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.

        :return: elapsed time, in seconds.
        :see: `finetune`
        """
        print "-- Beginning of fine-tuning (%d epochs) --" % (epochs)
        delta = self.finetune(x_train, y_train, batch_size, learning_rate, epochs, growth_factor, growth_threshold, verbose)
        print "-- End of fine-tuning (lasted %f minutes) --" % (delta/60.)
        return delta

class DeepNeuralNetwork(MultiLayerPerceptron):
    """
    A `DeepNeuralNetwork` (DNN) is a specialization of the MLP, where the
    layers are pretrained on the training examples (:math:`\mathbf{x})
    using a Stacked `AutoEncoder` strategy. It has been specifically designed
    for data that lies in a high-dimensional input space.

    :see: `MultiLayerPerceptron`, http://www.deeplearning.net/tutorial/SdA.html
    """
    def __init__(self, nUnitsInput, nUnitsOutput=[], outputActivation=Sigmoid):
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
        MultiLayerPerceptron.__init__(self, nUnitsInput + nUnitsOutput, outputActivation)
        self.nUnitsInput = nUnitsInput
        self.nUnitsOutput = nUnitsOutput[::-1] # reverse order

        # Construction des AutoEncoders
        self.inputAutoEncoders = []
        self.outputAutoEncoders = []

        x = T.matrix('x')
        y = T.matrix('y')
        for nInputs,nOutputs in zip(self.nUnitsInput, self.nUnitsInput[1:]):
            ae = AutoEncoder(nInputs, nOutputs)
            ae.linkInputs(x,nInputs)
            ae.prepare()
            ae.criterion = CrossEntropy(ae.outputs, y)
            self.inputAutoEncoders.append(ae)

        for nInputs,nOutputs in zip(self.nUnitsOutput, self.nUnitsOutput[1:]):
            ae = AutoEncoder(nInputs, nOutputs)
            ae.linkInputs(x,nInputs)
            ae.prepare()
            ae.criterion = CrossEntropy(ae.outputs, y)
            self.outputAutoEncoders.append(ae)

        self.linkLinear = self.modules[(len(self.nUnitsInput)-1)*2]
        self.linkData = []

    def prepareParams(self):
        if(self.modules):
            # Partage des paramètres en entrée du réseau
            for (ae, lin) in zip(self.inputAutoEncoders, self.modules[::2]):
                lin.prepareParams(ae.params[0], ae.params[1])
                self.params.extend(ae.params[0:2])
            # Partage des paramètres en sortie du réseau
            for (ae, lin) in zip(self.outputAutoEncoders, reversed(self.modules[::2])):
                lin.prepareParams(ae.params[0].T, ae.params[2])
                self.params.extend(ae.params[0:3:2])
            # Préparation de la couche de liaison
            self.linkLinear.prepareParams()
            self.params.extend(self.linkLinear.params)

    def pretrainAutoEncoders(self, data, autoEncoders, batch_size=1, learning_rate=1.0, epochs=100, growth_factor=1.25, growth_threshold=5, verbose=True):
        """
        Performs the unsupervised learning step of the autoencoders,
        using a batch-gradient backpropagation algorithm. This step can
        be applied to the autoencoders located on the input side or on
        the output side. Classically, in a `DeepNeuralNetwork`, only the
        input autoencoders are pretrained. The data used for this pretraining
        step can be the input training dataset used for the supervised learning
        (see `finetune`), or a subset of this dataset, or else a specially
        crafted input pretraining dataset.

        The `InputOutputDeepArchitecture` also pretrains the output autoencoders,
        in the same way the `DeepNeuralNetwork` does for input autoencoders. In
        this case, the given training data are the labels (:math:`\mathbf{y}`)
        and not the examples (:math:`\mathbf{x}`) (i.e. the labels that the network
        must predict).

        Once an `AutoEncoder` is learned, the projection layer is kept and used
        to initialize the network layers. The backprojection part is not useful
        anymore.

        The `learning_rate` is made adaptative with the `growth_factor` multiplier.
        If the mean loss a number is improved during `growth_threshold` successive epochs,
        then the `learning_rate` is increased, and if the mean loss is degraded during
        one epoch, then the `learning_rate` is decreased.

        :Parameters:
            data : :numpy:`ndarray`
                The training data (examples or labels).
            autoEncoders : `AutoEncoder` list
                The list of autoencoders used for training (input or output autoencoders).
            batch_size : int
                The number of training samples in each mini-batch.
            learning_rate : float
                The rate used to update the parameters with the gradient.
            epochs : int
                The number of epochs to run the training algorithm.
            growth_factor : float
                The multiplier factor used to increase or decrease the `learning_rate`.
            growth_threshold : float
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.
        """
        n_train_batches = data.shape[0]/batch_size
        inputs = data
        global_start_time = time.clock()
        for (ae,layer) in zip(autoEncoders, xrange(len(autoEncoders))):
            fn = ae.trainFunction(batch_size, learning_rate, True)
            if(verbose):
                print "Layer %d/%d" % (layer+1,len(autoEncoders))
                start_time = time.clock()
            mean_loss = float('inf')
            good_epochs = 0
            for epoch in xrange(epochs):
                if(verbose):
                    print "",
                c = []
                for i in xrange(n_train_batches):
                    loss = fn(inputs, inputs, i)
                    c.append(loss)
                    if(verbose):
                        print "\r  | |_Batch %d/%d, loss : %f" % (i+1, n_train_batches, loss),
                        sys.stdout.flush()

                new_mean_loss = np.mean(c)
                if(new_mean_loss < mean_loss):
                    good_epochs += 1
                    if(good_epochs >= growth_threshold):
                        good_epochs = 0
                        if(verbose):
                            print "\r# learning rate : %f > %f" % (learning_rate, learning_rate*growth_factor)
                        learning_rate = learning_rate*growth_factor
                        fn = ae.trainFunction(batch_size, learning_rate, True)
                else:
                    good_epochs = 0
                    if(verbose):
                        print "\r# learning rate : %f > %f" % (learning_rate, learning_rate/growth_factor)
                    learning_rate = learning_rate/growth_factor
                    fn = ae.trainFunction(batch_size, learning_rate, True)
                mean_loss = new_mean_loss
                if(verbose):
                    print "\r  |_Epoch %d, mean loss : %f" % (epoch, mean_loss)
            if(verbose):
                end_time = time.clock()
                print "Pre-learning layer %d took %f minutes" % (layer, (end_time - start_time)/60.)
            inputs = (ae.hiddenValues(inputs)+1)/2
        self.linkData.append(inputs)
        return (time.clock()-start_time)

    def train(self, x_train, y_train, batch_size=1, learning_rate=1.0, pretraining_learning_rate=2.0, epochs=100, growth_factor=1.25, growth_threshold=5, verbose=True):
        """
        Performs the pretraining step for the input autoencoders (`pretrainAutoEncoders`),
        and the supervised learning step (`finetune`).

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
            growth_threshold : float
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.

        :return: elapsed time, in seconds.
        :see: `pretrainAutoEncoders`, `finetune`
        """
        totalDelta = 0.0
        if(verbose):
            print "-- Beginning of input layers pre-training (%d epochs) --" % (epochs)
        delta = self.pretrainAutoEncoders(x_train, self.inputAutoEncoders, batch_size, pretraining_learning_rate, epochs, growth_factor, growth_threshold, verbose)
        totalDelta += delta
        if(verbose):
            print "-- End of input layers pre-training (lasted %f minutes) --" % (delta/60.)
            print "-- Beginning of fine-tuning (%d epochs) --" % (epochs)
        delta = self.finetune(x_train, y_train, batch_size, learning_rate, epochs, growth_factor, growth_threshold, verbose)
        totalDelta += delta
        if(verbose):
            print "-- End of fine-tuning (lasted %f minutes) --" % (delta/60.)
        return totalDelta


class InputOutputDeepArchitecture(DeepNeuralNetwork):
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

    def __init__(self, nUnitsInput, nUnitsOutput=[], outputActivation=Sigmoid):
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
        DeepNeuralNetwork.__init__(self, nUnitsInput, nUnitsOutput, outputActivation)

    def train(self, x_train, y_train, batch_size=1, learning_rate=1.0, pretraining_learning_rate=2.0, epochs=100, growth_factor=1.25, growth_threshold=5, verbose=True, pretrainLink=False):
        """
        Performs the pretraining step for the input and output autoencoders
        (`pretrainAutoEncoders`), optionally the semi-supervised pretraining step
        of the link layer (`pretrainLink`), and finally the supervised learning
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
            growth_threshold : float
                The number of successive loss-improving epochs after which the `learning_rate` must be updated.
            verbose : bool
                If true, information about the training process will be displayed on the standard output.

        :return: elapsed time, in seconds.
        :see: `pretrainAutoEncoders`, `pretrainLink`, `finetune`
        """
        totalDelta = 0.0
        if(verbose):
            print "-- Beginning of input layers pre-training (%d epochs) --" % (epochs)
        delta = self.pretrainAutoEncoders(x_train, self.inputAutoEncoders, batch_size, pretraining_learning_rate, epochs, growth_factor, growth_threshold, verbose)
        totalDelta += delta
        if(verbose):
            print "-- End of input layers pre-training (lasted %f minutes) --" % (delta/60.)
            print "-- Beginning of output layers pre-training (%d epochs) --" % (epochs)
        delta = self.pretrainAutoEncoders(y_train, self.outputAutoEncoders, batch_size, pretraining_learning_rate, epochs, growth_factor, growth_threshold, verbose)
        totalDelta += delta
        if(verbose):
            print "-- End of output layers pre-training (lasted %f minutes) --" % (delta/60.)
        if(pretrainLink):
            if(verbose):
                print "-- Beginning of link layer pre-training (%d epochs) --" % (epochs)
            delta = self.pretrainLink(batch_size, pretraining_learning_rate, epochs, growth_factor, growth_threshold, verbose)
            totalDelta += delta
            if(verbose):
                print "-- End of link layer pre-training (lasted %f minutes) --" % (delta/60.)

        if(verbose):
            print "-- Beginning of fine-tuning (%d epochs) --" % (epochs)
        delta = self.finetune(x_train, y_train, batch_size, learning_rate, epochs, growth_factor, growth_threshold, verbose)
        totalDelta += delta
        if(verbose):
            print "-- End of fine-tuning (lasted %f minutes) --" % (delta/60.)
        return totalDelta

    def pretrainLink(self, batch_size=1, learning_rate=1.0, epochs=100, growth_factor=1.25, growth_threshold=5, verbose=True):
        """
        Performs the semi-supervised learning step of the link layer (i.e. the
        layer situated between input and output layers). This layer is taken
        as a one-layer perceptron, and trained using as input and output data
        the hidden representations of the deepmost input and output autoencoders,
        respectively.

        The learning process is the exactly the same that the one used for the
        `MultiLayerPerceptron`.
        """

        x = T.matrix('x')
        y = T.matrix('y')
        nInputs = self.linkData[0].shape[1]
        nOutputs = self.linkData[1].shape[1]
        mlp = MultiLayerPerceptron([nOutputs], Tanh)
        mlp.linkInputs(x, nInputs)
        mlp.prepareGeometry()
        mlp.modules[0].prepareParams(self.linkLayer.params[0],self.linkLayer.params[1])
        mlp.params.extend(mlp.modules[0].params)
        mlp.prepareOutput()
        mlp.prepared = True
        mlp.criterion = MeanSquareError(mlp.outputs, y)
        return mlp.train(self.linkData[0], self.linkData[1], batch_size, learning_rate, epochs, growth_factor, growth_threshold, verbose)
