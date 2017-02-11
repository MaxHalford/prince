#!/usr/bin/python

from setuptools import setup, find_packages

from prince import  __author__, __license__, __title__, __version__


setup(
    author=__author__,
    author_email='maxhalford25@gmail.com',
    description='Factor analysis for in Python',
    install_requires=[
        'fbpca>=1.0',
        'matplotlib>=2.0',
        'numpy>=1.11.2',
        'pandas>=0.18.0',
        'scipy>=0.16.0'
    ],
    license=__license__,
    name=__title__,
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/Prince',
    version=__version__,
)
