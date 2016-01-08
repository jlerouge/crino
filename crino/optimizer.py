# -*- coding: utf-8 -*-

#	Copyright (c) 2014-2015 Soufiane Belharbi, Clément Chatelain,
#	Romain Hérault, Julien Lerouge, Romain Modzelewski (LITIS - EA 4108).
#	All rights reserved.
#
#	This file is part of Crino.
#
#	Crino is free software: you can redistribute it and/or modify
#	it under the terms of the GNU Lesser General Public License as published
#	by the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	Crino is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU Lesser General Public License for more details.
#
#	You should have received a copy of the GNU Lesser General Public License
#	along with Crino. If not, see <http://www.gnu.org/licenses/>.

"""
The `optimizer` module provides some optimization algorithms dedicated to
neural networks learning.

The currently implemented `Optimizer`s are :
	- `StochasticGradientDescent`
	- ...

See their respective documentations for more details about their use.

:see: `criterion`, `network`
"""

import sys
import datetime as DT
import numpy as np
import theano
import theano.tensor as T

from crino.criterion import Criterion

class Optimizer(object):
	"""
	An `Optimizer` is an algorithm able to train a neural network, using
	training data. It optimizes the parameters of the network to minimize some
	differentiable criterion. The criterion is used to compute the partial
	gradients of the parameters and perform a gradient descent.

	:attention: This is an abstract class, it must be derived to be used.
	"""
	def __init__(self, **params):
		"""
		Constructs a new `Optimizer` object.
		:Parameters:
			params : dict
				The learning parameters, encoded in a dictionary, that are used
				in the `finetune` method.

				Possible keys: batch_size, learning_rate, epochs, growth_factor,
				growth_threshold, badmove_threshold, verbose.
		"""
		(params, unknown) = self.checkLearningParameters(params)
		if len(unknown) > 0:
			warnings.warn("Unknown training parameters: %s." % (unknown))
		params = self.defaultLearningParameters(params)
		for key in params:
			setattr(self, key, params[key])

	def initEpochHook(self,finetune_vars):
		pass

	def checkEpochHook(self,finetune_vars):
		return False

	def initBadmoveHook(self,finetune_vars):
		pass

	def checkBadmoveHook(self,finetune_vars):
		return False

	def initBatchHook(self,finetune_vars):
		pass

	def checkBatchHook(self,finetune_vars):
		return False

	def checkLearningParameters(self,params):
		known = {}
		unknown = {}
		known_keys = self.defaultValues.keys()
		for (key, value) in params.items():
			if key in known_keys:
				known[key] = value
			else:
				unknown[key] = value
		return (known,unknown)

	def defaultLearningParameters(self,params):
		default = dict(params)
		for key in self.defaultValues.keys():
			if not(params.has_key(key)):
				default[key] = self.defaultValues[key]
		return default

	def optimize(self, module, x_train, y_train):
		raise NotImplementedError("This class must be derived.")

class StochasticGradientDescent(Optimizer):
		def __init__(self, **params):
			"""
			Constructs a new `StochasticGradientDescent` object.
			:Parameters:

			batch_size : int
				The size of the batches to use for gradient descent :
					- 1 for stochastic gradient descent;
					- :math:`n \in ]1..N_{train}[` for mini-batch gradient descent (:math:`N_{train}` must be a multiple of n);
					- :math:`N_{train}` for batch gradient descent.

				(:math:`N_{train}` is the total number of training examples)
			lr : float
				The learning rate.
			"""
			self.defaultValues = {'batch_size':1, 'learning_rate':1.0, 'epochs':100, 'growth_factor':1.25, 'growth_threshold':5, 'badmove_threshold':10, 'verbose':True}
			Optimizer.__init__(self, **params)

		def optimize(self, module, x_train, y_train):
			"""
			Performs the supervised learning step of the `MultiLayerPerceptron`,
			using a batch-gradient backpropagation algorithm. The `learning_rate`
			is made adaptative with the `growth_factor` multiplier. If the mean loss
			is improved during `growth_threshold` successive epochs, then the
			`learning_rate` is increased. If the mean loss is degraded, the epoche
			is called a "bad move", and the `learning_rate` is decreased until the
			mean loss is improved again. If the mean loss cannot be improved within
			`badmove_threshold` trials, then the last trained parameters are kept
			even though, and the finetuning goes further.

			:Parameters:
				x_train : :theano:`SharedVariable` from :numpy:`ndarray`
					The training examples.
				y_train : :theano:`SharedVariable` from :numpy:`ndarray`
					The training labels.
			:return: elapsed time, in datetime.
			"""

			# Compilation d'une fonction theano pour l'apprentissage du modèle
			train = module.trainFunction()
			hold = module.holdFunction()
			restore = module.restoreFunction()
			criterion = module.criterionFunction()

			nBatches = x_train.shape[0]/self.batch_size
			lr = self.learning_rate
			globalStartTime = DT.datetime.now()
			meanLoss = criterion(x_train, y_train)
			goodEpochs = 0
			self.fullHistory = [(-1,lr,meanLoss)]
			self.history = [meanLoss]

			self.initEpochHook(locals())
			for epoch in xrange(self.epochs):
				epochStartTime = DT.datetime.now()
				lossByBatch = []
				hold()
				if(self.verbose):
					print "",

				self.initBadmoveHook(locals())
				for badmoves in xrange(self.badmove_threshold):

					self.initBatchHook(locals())
					for lbatch_index in xrange(nBatches):
						loss = train(x_train, y_train, lr)
						lossByBatch.append(loss)
						if(self.verbose):
							print "\r  | |_Batch %d/%d, loss : %f" % (lbatch_index+1, nBatches, loss),
							sys.stdout.flush()
						if self.checkBatchHook(locals()):
							break

					meanLossNew = np.mean(lossByBatch)
					self.fullHistory.append((epoch,lr,meanLossNew))

					if self.checkBadmoveHook(locals()):
						break

					if  meanLossNew < meanLoss:
						goodEpochs +=  1
						break

					if (badmoves+1 < self.badmove_threshold):
						if(self.verbose):
							print "\r# Bad move %f < %f; Learning rate : %f --> %f" % (meanLoss, meanLossNew, lr, lr/self.growth_factor)
						restore()
						lr = lr/self.growth_factor
					else:
						if(self.verbose):
							print("\r# Break Epoch on bad move threshold")
					goodEpochs = 0

				meanLoss = meanLossNew
				self.history.append(meanLoss)

				if(goodEpochs >= self.growth_threshold):
					goodEpochs = 0
					if(self.verbose):
						print "\r# Fast Track; Learning rate : %f > %f" % (lr, lr*self.growth_factor)
					lr = lr*self.growth_factor
				if(self.verbose):
					print "\r  |_Epoch %d/%d, mean loss : %f, duration (s) : %s" % (epoch+1, self.epochs, meanLossNew,(DT.datetime.now()-epochStartTime).total_seconds())

				if self.checkEpochHook(locals()):
					break

			return (DT.datetime.now()-globalStartTime)
