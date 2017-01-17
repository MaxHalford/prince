#!/usr/bin/python

from setuptools import setup, find_packages

from prince import  __author__, __license__, __title__, __version__


setup(
    author=__author__,
    author_email='maxhalford25@gmail.com',
    description='Factor analysis for in-memory datasets in Python',
    install_requires=[
        'fbpca>=1.0',
        'matplotlib>=1.5',
        'pandas>=0.18.0'
    ],
    license=__license__,
    name=__title__,
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/Prince',
    version=__version__,
)
