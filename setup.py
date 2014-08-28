from setuptools import setup


DISTNAME            = 'crino'
DESCRIPTION         = "Crino: a neural-network library based on Theano"
LONG_DESCRIPTION    = """Crino is a neural-network library based on Theano, that lets you hand-craft neural-network architectures using a modular framework. It also provides standard implementations for common architectures, such as MLP, DNN, and IODA which is our contribution to the neural-network community."""
AUTHOR          	= 'Julien Lerouge'
AUTHOR_EMAIL    	= 'julien.lerouge@litislab.fr'
URL                 = "https://github.com/jlerouge/crino"
LICENSE             = "LGPL"
VERSION             = "1.0.0"

classifiers =  ['Development Status :: 4 - Beta',
                'Programming Language :: Python',
                'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Operating System :: OS Independent']

if __name__ == "__main__":
setup(
	name = DISTNAME,
  	version = VERSION,
	author = AUTHOR,
	author_email = AUTHOR_EMAIL,
    description = DESCRIPTION,
    license = LICENSE,
    url = URL,
    long_description = LONG_DESCRIPTION,
    packages = ['crino'],
    classifiers =classifiers,
    install_requires=['theano','numpy','scipy']
)
