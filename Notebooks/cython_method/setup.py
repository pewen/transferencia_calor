from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("explicit_cython2", ["explicit_cython2.pyx"])]

setup(
  name = 'Explicit method using Cython',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()], 
  ext_modules = ext_modules
)

'''
ext_module = Extension(
    "explicit_cython2",
    ["explicit_cython2.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
'''