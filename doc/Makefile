all:
	make -C images/
	epydoc --debug -v --external-api=theano --external-api-file=theano:api-theano.txt --external-api-root=theano:http://deeplearning.net/software/theano/library/ --external-api=numpy --external-api-file=numpy:api-numpy.txt --external-api-root=numpy:http://docs.scipy.org/doc/numpy/reference/ --docformat restructuredtext --dotpath /usr/bin/dot --graph classtree -o html ../crino
	
clean:
	make -C images/ clean
	rm -rf html
