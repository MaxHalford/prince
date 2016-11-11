#!/usr/bin/python

from setuptools import setup, find_packages

from prince import __version__, __authors__, __project__


setup(
    author=__authors__,
    author_email=['axel.bellec@outlook.fr', 'maxhalford25@gmail.com'],
    dependency_links=[],
    description='Factorial analysis in Python',
    install_requires=open('setup/requirements.txt').read().splitlines(),
    license='MIT',
    name=__project__,
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/Prince',
    version=__version__,
)
