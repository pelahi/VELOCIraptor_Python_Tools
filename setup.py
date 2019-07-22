from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext
import sys

setup(
    name='VELOCIraptor_Python_Tools',
    ext_modules = cythonize("velociraptor_python_tools_cython.pyx",language_level=sys.version_info[0])
)

if(sys.version_info[0]==2):
    print("\033[93mDeprecationWarning:\033[0m This code was built with python 2 which will soon be no longer maintained, please consider moving to python 3")
