""" setup script """
import os
import platform

from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import numpy as np


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


compile_arguments = build_extra_compile_args()

algorithms = [
    ("uniparse.decoders.eisner", "uniparse/decoders/eisner.pyx"),
    ("uniparse.decoders.cle", "uniparse/decoders/cle.pyx"),
    ("uniparse.models.mst_encode", "uniparse/models/mst_encode.pyx")
]
extensions = []
for name, location in algorithms:
    e = Extension(
        name=name, 
        sources=[location], 
        extra_compile_args=compile_arguments,
        include_dirs=[np.get_include()],
        language='c++'
    )
    extensions.append(e)



with open("README.md", "rb") as f:
    README = f.read().decode("utf-8")

setup(
    name="UniParse",
    version=0.1,
    description="Universal graph based dependency parsing prototype framework",
    long_description=README,
    author="NLP group @ the IT University of Copenhagen",
    author_email="djam@itu",
    url="https://github.com/ITUnlp/UniParse",
    install_requires=['numpy', 'scipy', 'sklearn', 'tqdm', 'cython'],
    cmdclass={ 'build_ext': build_ext },
    ext_modules=extensions
    # entry_points={
    #     "console_scripts": [
    #         "ketl=ketl.cli:main",
    #     ],
    # })
)