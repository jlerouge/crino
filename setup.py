import crino
from setuptools import setup

DISTNAME            = 'crino'
AUTHOR              = 'Julien Lerouge'
AUTHOR_EMAIL        = 'julien.lerouge@litislab.fr'
DESCRIPTION         = "Crino: a neural-network library based on Theano"
LONG_DESCRIPTION    = """Crino is a neural-network library based on Theano, that lets you hand-craft neural-network architectures using a modular framework. It also provides standard implementations for common architectures, such as MLP, DNN, and IODA which is our contribution to the neural-network community."""
LICENSE             = "LGPL"
KEYWORDS            = "crino theano mlp dnn neural network machine learning artificial intelligence" 
URL                 = "http://julien.lerouge.me/crino/"
CLASSIFIERS =  ['Development Status :: 4 - Beta',
                'Programming Language :: Python',
                'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Operating System :: OS Independent']

if __name__ == "__main__":
	setup(
		name = DISTNAME,
  		version = crino.__version__,
		author = AUTHOR,
		author_email = AUTHOR_EMAIL,
		description = DESCRIPTION,
		long_description = LONG_DESCRIPTION,
		license = LICENSE,
		keywords = KEYWORDS,
		url = URL,
		packages = ['crino'],
		classifiers =CLASSIFIERS,
		install_requires=['numpy','scipy','theano']
	)
