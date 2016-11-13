#!/usr/bin/python

from setuptools import setup, find_packages

from prince import __version__, __author__, __project__


setup(
    author=__author__,
    author_email='maxhalford25@gmail.com',
    dependency_links=[],
    description='Factorial analysis in Python',
    install_requires=[
        'fbpca>=1.0',
        'matplotlib>=1.5',
        'pandas>=0.19.0'
    ],
    license='MIT',
    name=__project__,
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/Prince',
    version=__version__,
)
