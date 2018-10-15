""" 
setup script 
run 'python setup.py build_ext --inplace' from the root directory
"""
import os
import platform
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize

def build_extra_compile_args():
    args = ['-O3', "-ffast-math", "-march=native"]

    system_name = platform.system()
    if system_name == "Darwin":
        args.append("-stdlib=libc++")
    elif system_name == "Linux":
        args.append("-std=c++0x")
    else:
        raise ValueError("dno what windows/other is")

    return args


c_plusplus_extra_args = build_extra_compile_args()

extensions = [
    Extension(
        name="uniparse.decoders.eisner",
        sources=["uniparse/decoders/eisner.pyx"],
        extra_compile_args=['-O3'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="uniparse.decoders.cle",
        sources=["uniparse/decoders/cle.pyx"],
        extra_compile_args=['-O3'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="uniparse.models.mst_encode",
        sources=["uniparse/models/mst_encode.pyx"],
        language="c++",
        extra_compile_args=c_plusplus_extra_args,
        include_dirs=[np.get_include()]
    )
]

setup(ext_modules=cythonize(extensions))

# remove c and build folder
os.system("rm uniparse/decoders/*.c")
os.system("rm uniparse/models/*.cpp")
os.system("rm -fr ./build/")