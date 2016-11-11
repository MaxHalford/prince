#!/usr/bin/python

from setuptools import setup, find_packages

from prince import __version__, __authors__, __project__


setup(
    author=__authors__,
    author_email=['axel.bellec@outlook.fr', 'maxhalford25@gmail.com'],
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
