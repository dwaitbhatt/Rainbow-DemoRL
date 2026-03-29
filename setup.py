#!/usr/bin/env python3
"""
Setup script for Rainbow-DemoRL package.
"""

from setuptools import setup, find_packages

setup(
    name="rainbow_demorl",
    version="0.1.0",
    description="Rainbow-DemoRL",
    author="Dwait Bhatt",
    packages=find_packages(),
    install_requires=[
        "mani_skill==3.0.0b22",
        "torchrl>=0.9.2",
        "tyro>=0.9.7",
        "tensorboard>=2.18.0",
        "wandb>=0.19.2",
        "numpy>=1.22,<2"
    ],
    python_requires=">=3.9",
)