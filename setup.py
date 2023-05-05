from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
          name="edge_distance_aabb", 
          sources=["edge_distance_aabb.pyx"],
          depends=["edge_distance_aabb.h"],
          include_dirs=[numpy.get_include()]
)

setup(name='edge_distance_aabb',
      packages=find_packages(),
      ext_modules=cythonize([ext])
)
