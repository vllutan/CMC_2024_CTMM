from setuptools import Extension, setup
#from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "verle_cython",
        ["verle_cython.pyx"],
        extra_compile_args = ['/openmp']
    )    
]

setup(
      name = 'verle_cython.pyx',
      ext_modules = cythonize(ext_modules),
      include_dirs = [numpy.get_include()]
)