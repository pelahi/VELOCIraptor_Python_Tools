from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext

setup(
    name='VELOCIraptor_Python_Tools',
    ext_modules = cythonize("velociraptor_python_tools_cython.pyx")
)
