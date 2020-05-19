from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = [Extension("pyexamples", ["pyexamples.pyx", "examples.cpp"], language="c++")]
setup(
    name="pyexamples",
    ext_modules=cythonize(examples_extension, language_level=3)
)
