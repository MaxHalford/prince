#!/usr/bin/python

from setuptools import setup, find_packages


VERSION = '0.3.0'

setup(
    author='Max Halford',
    author_email='maxhalford25@gmail.com',
    description='Statistical factor analysis in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'fbpca==1.0',
        'matplotlib==2.2.2',
        'numpy==1.14.0',
        'pandas==0.22.0',
        'scipy==1.0.1'
    ],
    license='MIT',
    name='prince',
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/prince',
    version=VERSION,
)
