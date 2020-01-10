#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="nnest",
    version="0.3.0",
    description="Neural network nested sampling",
    author="Adam Moss",
    author_email="adam.moss@nottingham.ac.uk",
    maintainer="Adam Moss",
    maintainer_email="adam.moss@nottingham.ac.uk",
    url="https://github.com/adammoss/nnest/",
    license="MIT",
    packages=find_packages(),
    provides=["nnest"],
    install_requires=["torch>=1.3.1",
              "tensorboard>=1.14",
              "numpy",
              "scipy",
              "matplotlib",
              "pandas",
              "scikit-learn",
              "tqdm",
              "pillow",
              "getdist"
            ],
)