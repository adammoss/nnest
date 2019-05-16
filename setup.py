#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name = "nnest",
    version = "0.1.2",
    description = "Neural network nested sampling",
    author = "Adam Moss",
    author_email = "adam.moss@nottingham.ac.uk",
    maintainer = "Adam Moss",
    maintainer_email = "adam.moss@nottingham.ac.uk",
    url = "https://github.com/adammoss/nnest/",
    license = "MIT",
    packages = find_packages(),
    provides = ["nnest"],
    requires = ["torch", "tensorflow", "tensorboardX", "numpy", "scipy", "matplotlib", "pandas",
                "scikitlearn", "tqdm", "pillow"],
)
