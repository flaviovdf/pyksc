# Simple makefile

PYTHON ?= python
NOSETESTS ?= nosetests

all: clean build

build:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build/
	rm -rf src/build/
	find . -name "*.pyc" | xargs rm -f
	find . -name "*.c" | xargs rm -f
	find . -name "*.so" | xargs rm -f

test: clean build
	$(NOSETESTS)

trailing-spaces: 
	find -name "*.py" | xargs sed 's/^M$$//'
