from distutils.core import setup
from Cython.Build import cythonize
 
setup(
    ext_modules = cythonize("velociraptor_python_tools.pyx")
)
