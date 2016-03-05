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
**Crino: a neural-network library based on Theano**

Crino is an open-source `Python <http://www.python.org/>`_ library aimed at
building and training artificial neural-networks. It has been developed on top
of `Theano <http://deeplearning.net/software/theano/>`_, by researchers from the
`LITIS <http://www.litislab.eu>`_ laboratory. It helps scientists and/or
programmers to design neural-network architectures adapted to their needs, using
a modular framework inspired by Torch. Our library also provides vanilla
implementations, and learning algorithms, for these architectures :

    - auto-encoders (AE)
    - multi-layer perceptrons (MLP)
    - deep neural networks (DNN)
    - input-output deep architectures (IODA)

IODA is an extension of DNN architectures, which is useful in cases where both
input and output spaces are high-dimensional, and where there are strong
interdependences between output labels. The input and output layers of a IODA
are initialized with an unsupervised pre-training step, based on the stacked
auto-encoder strategy, commonly used in DNN training algorithms. Then, the
backpropagation algorithm performs the final supervised learning step.

Crino and IODA are research topics of the `Deep in Normandy
<http://deep.normastic.fr/>`_ research program, which is a `NVIDIA GPU Research
Center <https://developer.nvidia.com/academia/centers/normastic>`_.

:see: `GitHub repository <https://github.com/jlerouge/crino>`_,
      `Project homepage <http://julien.lerouge.me/crino/>`_
"""

# Init file for crino
__version__ = "0.2.0"
import crino.criterion
import crino.module
import crino.network
