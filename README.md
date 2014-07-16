# Crino: a neural-network library based on Theano

## Description
Crino is an open-source Python library aimed at building and training artificial neural-networks. It has been developed on top of [Theano](http://deeplearning.net/software/theano/), by researchers from the [LITIS laboratory](http://www.litislab.eu).

Crino lets you "hand-craft" neural-network architectures, using a modular framework inspired by [Torch](http://torch5.sourceforge.net/manual/nn/). Our library also provides standard implementations for :
* auto-encoders (AE)
* multi-layer perceptrons (MLP)
* deep neural networks (DNN)
* input-output deep architectures (IODA)

Crino is natively compatible with Matlab-like data, but you can easily adapt it to your needs using SciPy and NumPy.

## IODA
IODA is a specialization of the DNNs, specifically designed for cases where you have to deal with high-dimensional input and output spaces. The input and output layers are initialized with an unsupervised pre-training step. Then, the backpropagation algorithm performs the supervised learning final step. This process is based on the stacked auto-encoder strategy, commonly used by DNN training algorithms.

We are writing an article on IODA, we'll inform you as soon as it is ready to publish.

## Getting started
* Install Crino :
```bash
cd to/your/preferred/path
git clone https://github.com/jlerouge/crino.git
cd crino
sudo python setup.py install
```

* Run the given example :
```bash
cd example
chmod +x example.py
./example.py
```
* Check the [docs](http://jlerouge.github.io/soft/crino/)

## Credits
Crino is based on Theano :
* J. Bergstra, O. Breuleux, F. Bastien, P. Lamblin, R. Pascanu, G. Desjardins, J. Turian, D. Warde-Farley and Y. Bengio. [“Theano: A CPU and GPU Math Expression Compiler”](http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf). Proceedings of the Python for Scientific Computing Conference (SciPy) 2010. June 30 - July 3, Austin, TX

## FAQ
* **What does "device gpu is not available" mean ?**
    Your GPU card may not be compatible with CUDA technology (check http://www.geforce.com/hardware/technology/cuda/supported-gpus). If so, there is nothing to do. Otherwise, your theano installation may have a  problem (see http://deeplearning.net/software/theano/install.html#using-the-gpu).
* **Where does the name "Crino" come from ?**
    We developed this library as an extension of Theano. In Greek mythology, Crino is the daughter of Theano.

## Disclaimer
Copyright (c) 2014 Clément Chatelain, Romain Hérault, Benjamin Labbé, Julien Lerouge, Romain Modzelewski, LITIS - EA 4108.
All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published 
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

The sample data (located in `example/data`) are free of use.


