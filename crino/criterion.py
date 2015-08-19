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

"""
provides some differentiable loss functions in order to
perform a gradient descent on a hand-crafted neural network.
"""

import theano.tensor as T

class Criterion:
    """
    The Criterion class handles the loss computation between **ŷ** (the `outputs` vector of a `Module`)
    and **y** (the `targets` vector). This loss has to be differentiable, in order to perform a gradient descent.

    :attention: This is an abstract class, it must be derived to be used.
    """
    def __init__(self, outputs, targets):
        """
        Constructs a new `Criterion` object.

        :Parameters:
            outputs : :theano:`TensorVariable`
                The symbolic outputs vector of a network
            targets : :theano:`TensorVariable`
                The symbolic targets vector
        """

        self.outputs = outputs
        """
        :ivar: The symbolic outputs vector of the network, denoted :math:`\mathbf{\hat{y}}`.
        :type: :theano:`TensorVariable`
        """

        self.targets = targets
        """
        :ivar: The symbolic targets vector, denoted :math:`\mathbf{y}`, that will be estimated by the `outputs`.
        :type: :theano:`TensorVariable`
        """

        self.expression = None
        """
        :ivar: The symbolic expression that expresses the loss between the `outputs` and the `targets` vectors.
        :type: :theano:`TensorVariable`
        """

        self.prepare()

    def prepare(self):
        """
        Computes the symbolic expression of the loss.

        :attention: It must be implemented in derived classes.
        """
        raise NotImplementedError("This class must be derived.")


class CrossEntropy(Criterion):
    """
    The cross-entropy criterion is well suited for targets vector
    that are normalized between 0 and 1, used along with a final
    `Sigmoid` activation module.

    It has been experimentally demonstrated that an `AutoEncoder`
    trains faster with a cross-entropy criterion.

    The cross-entropy loss can be written as follows :

    :math:`L_{CE} = -1/N \ \sum_{k=1}^{N} (y\cdot log(\hat{y}) + (1-y)\cdot log(1-\hat{y}))`
    """

    def __init__(self, outputs, targets):
        """
        Constructs a new `CrossEntropy` criterion.

        :Parameters:
            outputs : :theano:`TensorVariable`
                The symbolic `outputs` vector of the network
            targets : :theano:`TensorVariable`
                The symbolic `targets` vector
        """
        Criterion.__init__(self, outputs, targets)

    def prepare(self):
        """ Computes the cross-entropy symbolic expression. """
        self.expression = -T.mean(self.targets*T.log(self.outputs) + (1-self.targets)*T.log(1-self.outputs))


class MeanSquareError(Criterion):
    """
    The mean square error criterion is used in the least squares method,
    it is well suited for data fitting.

    The mean square loss can be written as follows :

    :math:`L_{MSE} = -1/N \ \sum_{k=1}^{N} (y-\hat{y})^2`
    """

    def __init__(self, outputs, targets):
        """
        Constructs a new `MeanSquareError` criterion.

        :Parameters:
            outputs : :theano:`TensorVariable`
                The symbolic `outputs` vector of the network
            targets : :theano:`TensorVariable`
                The symbolic `targets` vector
        """
        Criterion.__init__(self, outputs, targets)

    def prepare(self):
        """ Computes the mean square error symbolic expression. """
        self.expression = T.mean(T.sqr(self.outputs - self.targets))


class MeanAbsoluteError(Criterion):
    """
    The mean absolute error criterion is used in the least absolute deviations method.

    The mean absolute loss can be written as follows :

    :math:`L_{MSE} = -1/N \ \sum_{k=1}^{N} |y-\hat{y}|`
    """

    def __init__(self, outputs, targets):
        """
        Constructs a new `MeanAbsoluteError` criterion.

        :Parameters:
            outputs : :theano:`TensorVariable`
                The symbolic `outputs` vector of a network
            targets : :theano:`TensorVariable`
                The symbolic `targets` vector
        """
        Criterion.__init__(self, outputs, targets)

    def prepare(self):
        """ Computes the mean absolute error symbolic expression. """
        self.expression = T.mean(T.abs_(self.outputs - self.targets))

class NegativeLogLikelihood(Criterion):
    """
    The Negative Log Likelihood criterion is used in classification task.

    It is meant to be connected to a Softmax Module as the last layer of the MLP.

    The output size should be the same as the number of possible classes.

    The target size should be the same as the number of possible classes, with value in between [0..1].
    If not weighted, all 0 except 1 on the correct class.

    The Negative Log Likelihood can be written as follows :

    :math:`L_{NLL} = -log(<\hat{y},y>)`
    """

    def __init__(self, outputs, targets):
        """
        Constructs a new `NegativeLogLikelihood` criterion.

        :Parameters:
            outputs : :theano:`TensorVariable`
                The symbolic `outputs` vector of a network
            targets : :theano:`TensorVariable`
                The symbolic `targets` vector
        """
        Criterion.__init__(self, outputs, targets)

    def prepare(self):
        """ Computes the Negative Log Likelihood symbolic expression. """
        self.expression = -T.log(T.sum(self.outputs*self.targets))
