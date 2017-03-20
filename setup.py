#!/usr/bin/python

from setuptools import setup, find_packages


setup(
    author='Max Halford',
    author_email='maxhalford25@gmail.com',
    description='Factor analysis for in Python',
    install_requires=[
        'fbpca>=1.0',
        'matplotlib>=2.0',
        'numpy>=1.11.2',
        'pandas>=0.18.0',
        'scipy>=0.16.0'
    ],
    license='MIT',
    name='prince',
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/Prince',
    version='0.2.6',
)
