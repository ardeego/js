PYX_FILENAME = manhattanClusteringCy

all:
	python setup.py build_ext --inplace
annotate:
	cython -a $(PYX_FILENAME).pyx  
	firefox $(PYX_FILENAME).html
clean:
	rm -rf build
	rm *.c *.so
	rm *.c *.so
