#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name = "nnest",
    version = "0.1.0",
    description = "Neural network nested sampling",
    author = "Adam Moss",
    author_email = "adam.moss@nottingham.ac.uk",
    maintainer = "Adam Moss",
    maintainer_email = "adam.moss@nottingham.ac.uk",
    url = "https://github.com/adammoss/nnest/",
    license = "MIT",
    packages = ["nnest"],
    provides = ["nnest"],
    requires = ["torch", "tensorflow", "tensorboardX", "numpy", "scipy", "matplotlib", "pandas",
                "scikitlearn", "tqdm", "pillow"],
)
