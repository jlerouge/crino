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
The `module` module provides a modular architecture to build a neural network.

These modules can be either `Standalone` modules, i.e. modules that doesn't have
submodules, or `Container` modules, i.e. modules that are composed of one or
several other module(s). The `Container` modules are typically used to create
arbitrarily complex neural network architectures, while `Standalone` modules
acts as `Linear` regression layers or non-linear `Activation` layers.

The currently implemented `Standalone` modules are :
	- `Linear`
	- `Sigmoid`
	- `Softmax`
	- `Tanh`

The currently implemented `Container` modules are :
	- `Concat`
	- `Sequential`

See their respective documentations for more details about their use.

:see: `criterion`, `network`
"""

import theano
import theano.tensor as T
import numpy as np
import warnings
import cPickle as pickle
from crino.criterion import Criterion
from theano.compile.sharedvalue import SharedVariable

def load(filename):
	"""
	Loads and returns a `Module` previously saved with `Module.save` function.

	:Parameters:
		filename : str
			The path to the saved module.
	"""
	return pickle.load(open(filename, 'rb'))

class Module:
	"""
	A `Module` is a part of a neural network architecture,
	that may have parameters. Provided an input vector of
	fixed size, a module is able to compute an output vector,
	which size is specified at construction.
	According to their characteristics, several modules can
	be combined to build a new module.

	If a criterion is given to a module, it is able to compute
	the partial gradients of its parameters, in order to perform
	a gradient descent.

	:attention: This is an abstract class, it must be derived to be used.
	"""
	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `Module` object.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
		"""

		self.inputs = None
		"""
		:ivar: The symbolic `inputs` vector of the module, denoted :math:`\mathbf{x}`
		:type: :theano:`TensorVariable`
		"""

		self.outputs = None
		"""
		:ivar: The symbolic `outputs` vector of the module, denoted :math:`\mathbf{\hat{y}}`
		:type: :theano:`TensorVariable`
		"""

		self.nInputs = nInputs
		"""
		:ivar: The `inputs` size
		:type: int
		"""

		self.nOutputs = nOutputs
		"""
		:ivar: The `outputs` size
		:type: int
		"""

		self.params = []
		"""
		:ivar: The list of parameters
		:type: list
		"""

		self.updates = []
		"""
		:ivar: The list of state updates
		:type: list
		"""

		self.backupParams = []
		"""
		:ivar: The list of backup parameters
		:type: list
		"""

		self.prepared = False
		"""
		:ivar: Indicates whether the module have already been prepared
		:type: bool
		"""

	def linkModule(self, previous):
		"""
		Links the outputs of the previous module to the inputs of the current module.

		:Parameters:
			previous : `Module`
				The previous module to be linked with the current module.
		"""
		if self.prepared and (self.nInputs != previous.nOutputs):
			raise Exception("This module has already been prepared, you can't change its inputs size.")
		elif not(previous.outputs) :
			raise Exception("The inputs module has not been prepared before.")
		else :
			self.nInputs = previous.nOutputs
			self.inputs = previous.outputs


	def setInputs(self, vector, nInputs):
		"""
		Sets the symbolic vector as the inputs of the current module.

		:attention: You can't change the module inputs size once it has been `prepare()`'d

		:Parameters:
			vector : :theano:`TensorVariable`
				The symbolic vector that will serve as inputs.
			nInputs : int
				The size of this inputs vector.
		"""
		if self.prepared and (self.nInputs != nInputs):
			raise Exception("This module has already been prepared, you can't change its inputs size.")
		else:
			self.nInputs = nInputs
		self.inputs = vector

	def setCriterion(self, criterion):
		"""
		Sets the criterion for training, and compute the partial gradients
		of the parameters.

		:Parameters:
			vector : :theano:`TensorVariable`
				The symbolic vector that will serve as inputs.
			nInputs : int
				The size of this inputs vector.
		"""
		self.criterion = criterion
		self.gparams = T.grad(self.criterion.expression, self.params)

	def trainFunction(self, downcast=None):
		"""
		Constructs and compiles a Theano function in order to train the module.

		:Parameters:

			downcast : bool
				If true, allows the inputs data to be downcasted
				(e.g. from double to single precision floats for GPU use).
		:return: a Theano-function that performs one step of gradient descent
		:rtype: :theano:`function`
		"""

		# Définition des entrées
		lr = T.dscalar('lr')

		# Définition des mises à jour
		updates = []
		for param_i, grad_i in zip(self.params, self.gparams):
			updates.append((param_i, param_i - lr*grad_i))

		# Construction d'une fonction d'apprentissage theano
		return theano.function( inputs=[self.inputs, self.criterion.targets, lr],
								outputs=self.criterion.expression,
								updates=updates,
								allow_input_downcast=downcast)

	def train(self, x_train, y_train, optimizer):
		"""
		Performs the supervised learning step of the `MultiLayerPerceptron`.
		This function explicitly calls `finetune`, but displays a bit more information.

		:Parameters:
			x_train : :numpy:`ndarray`
				The training examples.
			y_train : :numpy:`ndarray`
				The training labels.
		:return: elapsed time, in datetime.
		:see: `finetune`
		"""
		if not (hasattr(self, 'params') and self.params):
			raise Exception("The module has no parameter, it cannot be trained.")
		if not (hasattr(self, 'criterion') and self.criterion):
			raise Exception("The module has no criterion, it cannot be trained.")
		if not (hasattr(self, 'gparams') and self.gparams):
			raise Exception("The partial gradients of the module have not been computed.")

		if(optimizer.verbose):
			print "-- Start training --"
		delta = optimizer.optimize(self, x_train, y_train)
		if(optimizer.verbose):
			print "-- End training (lasted %s) --" % (delta)
		return delta


	def criterionFunction(self, downcast=None):
		"""
		Constructs and compiles a Theano function in order to compute the criterion on a given set.

		:Parameters:
			downcast : bool
				If true, allows the inputs data to be downcasted
				(e.g. from double to single precision floats for GPU use).
		:return: a Theano-function that computes the criterion
		:rtype: :theano:`function`
		"""
		return theano.function( inputs=[self.inputs, self.criterion.targets],
								outputs=self.criterion.expression,
								allow_input_downcast=downcast)

	def forwardFunction(self, downcast=None):
		"""
		Constructs and compiles a Theano function in order to compute the forward on a given set.

		:Parameters:
			input : :numpy:`ndarray`
				The test example(s) on which the neural network will compute its
				outputs.
			downcast : bool
				If true, allows the inputs data to be downcasted
				(e.g. from double to single precision floats for GPU use).
		:return: a Theano-function that performs a forward step
		:rtype: :theano:`function`
		"""
		return theano.function(	inputs=[self.inputs],
								outputs=self.outputs,
								updates=self.updates,
								allow_input_downcast=downcast)

	def forward(self, x_test, downcast=None):
		"""
		Performs the forward step on the given test example.

		:Parameters:
			x_test : :numpy:`ndarray`
				The test example(s) on which the neural network will compute its
				outputs.
			downcast : bool
				If true, allows the inputs data to be downcasted (e.g. from
				double to single precision floats for GPU use).

		:return: the outputs of the neural network on the given test example(s)
		:rtype: :numpy:`ndarray`
		"""
		if not(self.prepared):
			raise Exception("The module must be prepared before doing a forward step.")
		f = self.forwardFunction(downcast=downcast)
		return f(x_test)

	def holdFunction(self):
		"""
		TODO
		"""
		if self.params and self.backupParams:
			# Définition des mises à jour
			updates = []
			for param_i, backup_param_i in zip(self.params, self.backupParams):
				updates.append((backup_param_i, param_i))

			# Construction d'une fonction de hold
			return theano.function(inputs=[], updates=updates)
		else:
			return None

	def restoreFunction(self):
		"""
		TODO
		"""
		if self.params and self.backupParams:
			# Définition des mises à jour
			updates = []
			for param_i, backup_param_i in zip(self.params, self.backupParams):
				updates.append((param_i, backup_param_i))

			# Construction d'une fonction de restore
			return theano.function(inputs=[], updates=updates)
		else:
			return None

	def prepare(self):
		"""
		Prepares the module before learning.

		:attention: The inputs must be linked before preparation.
		"""
		if not self.prepared:
			if not self.inputs:
				raise Exception('The inputs of this module has not been linked before preparation.')
			else:
				self.prepareGeometry()
				self.prepareParams()
				self.prepareBackup()
				self.prepareOutputs()
				self.prepareUpdates()
				self.prepared = True
		else:
			warnings.warn("This module is already prepared.")

	def prepareGeometry(self):
		"""
		Sets correctly the geometry (`nInputs` and `nOutputs`) of the potential submodules.

		:attention: It must be implemented in derived classes.
		"""
		raise NotImplementedError("This class must be derived.")

	def prepareParams(self):
		"""
		Initializes the `params` of the module and its potential submodules.

		:attention: It must be implemented in derived classes.
		"""
		raise NotImplementedError("This class must be derived.")

	def prepareBackup(self):
		"""
		Initializes the `backupParams` of the module and its potential submodules.
		"""
		#
		if self.params:
			for param_i in self.params:
				theShape=theano.function(inputs=[],outputs=param_i.shape)
				data=np.zeros(theShape(),dtype=theano.config.floatX)
				self.backupParams.append(theano.shared(value=data, name='backup_'+param_i.name, borrow=True))

	def prepareOutputs(self):
		"""
		Computes the symbolic `outputs` of the module in respect to its `inputs`.

		:attention: It must be implemented in derived classes.
		"""
		raise NotImplementedError("This class must be derived.")

	def prepareUpdates(self):
		"""
		Computes the updates that are applied at each forward step.

		:attention: It must be implemented in derived classes.
		"""
		raise NotImplementedError("This class must be derived.")

	def save(self, filename):
		"""
		Saves this `Module` to a file.

		:Parameters:
			filename : str
				The path where the module is to be saved.
		"""
		pickle.dump(self, open(filename, 'wb'), protocol=-1)

class Standalone(Module):
	"""
	A `Standalone` module computes its `outputs` without relying
	on other modules, i.e. it doesn't have any submodule.

	:attention: This is an abstract class, it must be derived to be used.
	"""

	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `Standalone` module.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
		"""
		Module.__init__(self, nOutputs, nInputs)

	def prepareGeometry(self):
		"""
		Do nothing, as a standalone module doesn't have submodules.
		"""
		pass


class Linear(Standalone):
	"""
	A `Linear` module computes its `outputs` as a linear transformation of its `inputs`.
	It has two `params` : :math:`W \in \mathcal{M}_{n_out \\times n_in}(\mathbb{R})`
	and :math:`b \in \mathbb{R}^{n_{out}}`.

	The `outputs` expression can be written as follows :
	:math:`\mathbf{\hat{y}} = W\\times \mathbf{x} + b`
	"""

	def __init__(self, nOutputs, nInputs=None, W_init=None, b_init=None):
		"""
		Constructs a new `Linear` module.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
			W_init : :numpy:`ndarray`
				The initialization matrix for W.
			b_init : :numpy:`ndarray`
				The initialization vector for b.
		"""

		Standalone.__init__(self, nOutputs, nInputs)

		self.W_init = W_init
		"""
		:ivar: The initialization matrix for `W`.
		:type: :numpy:`ndarray`
		"""

		self.b_init = b_init
		"""
		:ivar: The initialization vector for `b`.
		:type: :numpy:`ndarray`
		"""

		self.W = None
		"""
		:ivar: The symbolic linear transformation matrix.
		:type: :theano:`TensorVariable`
		"""

		self.b = None
		"""
		:ivar: The symbolic offset vector.
		:type: :theano:`TensorVariable`
		"""

	def prepareParams(self, W=None, b=None):
		"""
		Initializes the `W` and `b` `params` of the `Linear` module.

		:Parameters:
			W : :theano:`TensorVariable`
				If provided, the `Linear` module will use W as a shared parameter from another module.
			b : :theano:`TensorVariable`
				If provided, the `Linear` module will use b as a shared parameter from another module.

		:attention: `W_init` and `b_init` values will be ignored if existing W and b are passed, since they have already been initialized in another module.
		"""

		if W:
			self.W = W
		else:
			if (self.W_init == None):
				ext = np.sqrt(6./(self.nInputs + self.nOutputs))
				self.W_init = np.asarray(np.random.uniform(low=-ext,
								high=ext, size=(self.nInputs, self.nOutputs)),
								dtype=theano.config.floatX)
			self.W = theano.shared(value=self.W_init, name='W', borrow=True)

		if b:
			self.b = b
		else:
			if (self.b_init == None):
				self.b_init = np.zeros((self.nOutputs,), dtype=theano.config.floatX)
			self.b = theano.shared(value=self.b_init, name='b', borrow=True)

		self.params = [self.W, self.b]

	def prepareOutputs(self):
		"""
		Computes the linear relation :math:`\mathbf{\hat{y}} = W\\times \mathbf{x} + b`
		"""
		self.outputs = T.dot(self.inputs, self.W) + self.b


	def prepareUpdates(self):
		"""
		`Linear` module has no states, so it does nothing.
		"""
		pass

class LongShortTermMemory(Standalone):
	"""
	A `LongShortTermMemory` (LSTM) module computes its `outputs` as ...
	TODO
	"""

	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `LongShortTermMemory` (LSTM) module.
		TODO
		"""
		Standalone.__init__(self, nOutputs, nInputs)

	def initialWeights(self, low, high, size, name=None):
		W = np.random.uniform(low, high, size)
		W = np.asarray(W, dtype=theano.config.floatX)
		return theano.shared(value=W, name=name, borrow=True)

	def prepareParams(self):
		"""
		TODO
		"""
		self.W_xi = self.initialWeights(-1.0, 1.0, (self.nInputs, self.nOutputs), 'W_xi')
		self.W_hi = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_hi')
		self.W_ci = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_ci')
		self.b_i = self.initialWeights(-0.5, 0.5, (self.nOutputs, ), 'b_i')

		self.W_xf = self.initialWeights(-1.0, 1.0, (self.nInputs, self.nOutputs), 'W_xf')
		self.W_hf = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_hf')
		self.W_cf = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_cf')
		self.b_f = self.initialWeights(0.0, 1.0, (self.nOutputs, ), 'b_f')

		self.W_xc = self.initialWeights(-1.0, 1.0, (self.nInputs, self.nOutputs), 'W_xc')
		self.W_hc = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_hc')
		self.b_c = self.initialWeights(0.0, 0.0, (self.nOutputs, ), 'b_c')

		self.W_xo = self.initialWeights(-1.0, 1.0, (self.nInputs, self.nOutputs), 'W_xo')
		self.W_ho = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_ho')
		self.W_co = self.initialWeights(-1.0, 1.0, (self.nOutputs, self.nOutputs), 'W_co')
		self.b_o = self.initialWeights(-0.5, 0.5, (self.nOutputs, ), 'b_o')

		self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o]

	def prepareOutputs(self):
		"""
		TODO
		"""
		self.c_tm1 = theano.shared(value=np.zeros((self.nOutputs,1), dtype=theano.config.floatX), name='c_tm1')
		self.h_tm1 = theano.shared(value=np.zeros((self.nOutputs,1), dtype=theano.config.floatX), name='h_tm1')

		self.i_t = T.nnet.sigmoid(T.dot(self.inputs, self.W_xi) + T.dot(self.h_tm1, self.W_hi) + T.dot(self.c_tm1, self.W_ci) + self.b_i)
		self.f_t = T.nnet.sigmoid(T.dot(self.inputs, self.W_xf) + T.dot(self.h_tm1, self.W_hf) + T.dot(self.c_tm1, self.W_cf) + self.b_f)
		self.c_t = self.f_t * self.c_tm1 + self.i_t * T.tanh(T.dot(self.inputs, self.W_xc) + T.dot(self.h_tm1, self.W_hc) + self.b_c)
		self.o_t = T.nnet.sigmoid(T.dot(self.inputs, self.W_xo)+ T.dot(self.h_tm1, self.W_ho) + T.dot(self.c_t, self.W_co)  + self.b_o)
		self.h_t = self.o_t * T.tanh(self.c_t)
		self.outputs = self.h_t

	def prepareUpdates(self):
		"""
		Initializes the state `updates` of the LongShortTermMemory layer.
		"""
		self.updates.append((self.h_tm1, self.h_t))
		self.updates.append((self.c_tm1, self.c_t))

class Container(Module):
	"""
	A `Container` module computes its `outputs` thanks to
	other modules, i.e. it includes several submodules.
	The way these submodules are organized depends on the
	implementation.

	:attention: This is an abstract class, it must be derived to be used.
	"""

	def __init__(self, mods=[], nInputs=None):
		"""
		Constructs a new `Container` module.

		:Parameters:
			mods : list
				A list of submodules to add to the container.
			nInputs : int
				The `inputs` size.
		"""
		Module.__init__(self, None, nInputs)

		self.modules=[]
		"""
		:ivar: The list of submodules
		:type: list
		"""
		if mods:
			map(self.add, mods)

	def add(self, module):
		"""
		Adds a module to the `Container`.

		:Parameters:
			module : `Module`
				The submodule to add.
		"""
		self.modules.append(module)

	def prepareParams(self):
		"""
		Initializes the `params` of the submodules. The `Container` module
		`params` will include all the parameters of its submodules.
		"""
		if(self.modules):
			for mod in self.modules:
				mod.prepareParams()
				self.params.extend(mod.params)

	def prepareUpdates(self):
		"""
		Initializes the `updates` of the submodules. The `Container` module
		`updates` will include all the updates of its submodules.
		"""
		if(self.modules):
			for mod in self.modules:
				mod.prepareUpdates()
				self.updates.extend(mod.updates)

class Sequential(Container):
	"""
	A `Sequential` module computes its `outputs` sequentially,
	i.e. each `outputs` of its submodules is linked to the `inputs`
	of the following module. The number of submodules in the sequence
	can be chosen arbitrarily, but the `inputs` and `outputs` sizes
	must be the same throughout the sequence.

	.. image:: ../images/sequential.png

	"""
	def __init__(self, mods=[], nInputs=None):
		"""
		Constructs a new `Sequential` container.

		:Parameters:
			mods : list
				A list of submodules to add to the sequence. They will be linked in the same order as provided.
			nInputs : int
				The `inputs` size of the sequence (and of all the submodules).
		"""
		Container.__init__(self, mods, nInputs)

	def prepareGeometry(self):
		"""
		Sets the same `inputs` and `outputs` size for all submodules,
		and prepare their internal geometry.
		"""

		if(self.modules):
			# Le conteneur séquentiel a pour sortie le dernier module
			self.nOutputs = self.modules[-1].nOutputs
			# Le premier module a pour entrée celle du conteneur séquentiel
			first = self.modules[0]
			first.nInputs = self.nInputs
			first.prepareGeometry()
			for base, new in zip(self.modules, self.modules[1:]):
				new.nInputs = base.nOutputs
				new.prepareGeometry()

	def prepareOutputs(self):
		"""
		Computes sequentially the symbolic `outputs` of the module.
		"""
		if(self.modules):
			first = self.modules[0]
			first.inputs = self.inputs
			first.prepareOutputs()
			first.prepared = True
			for base, new in zip(self.modules, self.modules[1:]):
				new.linkModule(base)
				new.prepareOutputs()
				new.prepared = True
			self.outputs = self.modules[-1].outputs


class Concat(Container):
	"""
	A `Concat` module computes its `outputs` in parallel,
	i.e. it subdivides its `inputs` in :math:`n \in \mathbb{N}`
	parts of fixed sizes. The sum of the submodules `inputs`
	sizes must equal the total `inputs` size of the `Concat`.

	.. image:: ../images/concat.png

	"""
	def __init__(self, mods=[], nInputs=None):
		"""
		Constructs a new `Concat` container.

		:Parameters:
			mods : list
				A list of submodules to add to the concat. Each one will receive a part of the `inputs`.
			nInputs : int
				The `inputs` size of the concat.
		"""
		Container.__init__(self, mods, nInputs)

	def prepareGeometry(self):
		"""
		Sets the `outputs` size for the `Concat`, and prepare the submodules internal geometry.
		"""
		if(self.modules):
			# Le conteneur de concaténation concaténe les sorties de ses modules
			self.nOutputs = reduce(lambda x,y: x+y.nOutputs, self.modules, 0)
			# On vérifie que le partitionnement des entrées est correct
			nInputs = reduce(lambda x,y: x+y.nInputs, self.modules, 0)
			if (nInputs != self.nInputs):
				raise Exception("The total inputs sizes of the sub-modules is wrong.")

			map(lambda x:x.prepareGeometry(), self.modules)

	def prepareOutputs(self):
		"""
		Computes the symbolic `outputs` of all the submodules, and concatenate them to get the complete `outputs`.
		"""

		if(self.modules):
			temp = 0
			for mod in self.modules:
				mod.setInputs(self.inputs[:,temp:temp+mod.nInputs], mod.nInputs)
				temp += mod.nInputs
				mod.prepareOutputs()
				mod.prepared = True
			self.outputs = T.concatenate(map(lambda x: x.outputs, self.modules), axis=1)


class Activation(Standalone):
	"""
	An `Activation` module computes its `outputs` without any parameter, but with a function
	:math:`f \,:\, \mathbb{R}^n -> \mathbb{R}^n` applied to its `inputs` vector.
	This function is generally non-linear, because its purpose is to provide non-linearity
	to the neural network.
	"""

	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `Activation` module.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
		"""

		Standalone.__init__(self, nOutputs, nInputs)

	def prepareParams(self):
		"""
		`Activation` module has no `params`, so it does nothing.
		"""
		pass

	def prepareUpdates(self):
		"""
		`Activation` module has no states, so it does nothing.
		"""
		pass

class Tanh(Activation):
	"""
	A `Tanh` activation module computes its `outputs` with the
	non-linear element-wise hyperbolic tangent function, that can be defined as
	:math:`tanh(\mathbf{x}) = [\dfrac{exp(x_i)-exp(-x_i)}{exp(x_i)+exp(-x_i)}]_{i=1}^n`,
	with :math:`\mathbf{x} = [x_1, x_2, \dots, x_n] \in \mathbb{R}^n`.
	"""
	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `Tanh` activation module.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
		"""
		Activation.__init__(self, nOutputs, nInputs=None)

	def prepareOutputs(self):
		"""
		Computes the tanh function :math:`\mathbf{\hat{y}} = = [\dfrac{exp(x_i)-exp(-x_i)}{exp(x_i)+exp(-x_i)}]_{i=1}^n`
		"""
		self.outputs = T.tanh(self.inputs)

class Sigmoid(Activation):
	"""
	A `Sigmoid` activation module computes its `outputs` with the
	non-linear element-wise sigmoid function, that can be defined as
	:math:`\sigma(\mathbf{x}) = (1+tanh(\mathbf{x}/2))/2 = [1/(1+exp(-x_i))]_{i=1}^n`,
	with :math:`\mathbf{x} = [x_1, x_2, \dots, x_n] \in \mathbb{R}^n`.
	"""
	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `Sigmoid` activation module.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
		"""
		Activation.__init__(self, nOutputs, nInputs)

	def prepareOutputs(self):
		"""
		Computes the sigmoid function :math:`\mathbf{\hat{y}} = [1/(1+exp(-x_i))]_{i=1}^n`
		"""
		self.outputs = T.nnet.sigmoid(self.inputs)

class Softmax(Activation):
	"""
	A `Softmax` activation module computes its `outputs` with the
	non-linear softmax function, that can be defined as
	:math:`softmax(\mathbf{x}) = [exp(x_i)/\sum_{i=1}^n exp(x_i)]_{i=1}^n`,
	with :math:`\mathbf{x} = [x_1, x_2, \dots, x_n] \in \mathbb{R}^n`.
	"""

	def __init__(self, nOutputs, nInputs=None):
		"""
		Constructs a new `Softmax` activation module.

		:Parameters:
			nOutputs : int
				The `outputs` size.
			nInputs : int
				The `inputs` size.
		"""
		Activation.__init__(self, nOutputs, nInputs)

	def prepareOutputs(self):
		"""
		Computes the softmax function :math:`\mathbf{\hat{y}} = [exp(x_i)/\sum_{i=1}^n exp(x_i)]_{i=1}^n`
		"""
		self.outputs = T.nnet.softmax(self.inputs)
