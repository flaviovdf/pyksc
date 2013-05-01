PY-KSC
======

Implementation of the KSC time series clustering algorithm.
See [1]_ for details:

Dependencies for library
------------------------
   * Numpy
   * Cython

Dependencies for scripts
------------------------
   * Scipy
   * Matplotlib

How to install
--------------

Clone the repo

::

$ git clone https://github.com/flaviovdf/pyksc.git

Make sure you have cython and numpy. If not run as root (or use your distros package manager)

::

$ pip install numpy

::

$ pip install Cython

Install

::

$ python setup.py install

References
----------
.. [1] J. Yang and J. Leskovec, 
   "Patterns of Temporal Variation in Online Media" - WSDM'11  
   http://dl.acm.org/citation.cfm?id=1935863
