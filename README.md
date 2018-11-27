# This is a python toolkit for VELOCraptor and TreeFrog software packages

================================================================================================

 ## developed by:
    Pascal Jahan Elahi (continuously)
    Additional contributors:
    Rhys Poulton

================================================================================================

## Content
    velociraptor_python_tools.py    contains main source code
    setup.py                        used to build cythonized verison 

================================================================================================

The python tool kit can be used simply by updating the system path in python and import the code

import sys
sys.path.append('/dir/to/tools/')
import velociraptor_python_tools as vpt

There is also an associated setup to build the cythonized version. Simply run

python setup.py build_ext --inplace

This will build velociraptor_python_tools_cython which can be loaded via 

import velociraptor_python_tools_cython as vpt

