#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='torch_unet',
      author='Edoardo Holzl',
      version='0.1.0',
      packages=find_packages(where='.'),
      zip_safe=False)
