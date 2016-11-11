#!/usr/bin/python

"""Prince packaging instructions."""

from setuptools import setup, find_packages
from prince import __version__, __authors__, __project__

README = 'README.md'
REQUIREMENTS = 'setup/requirements.txt'


def get_requirements():
    """Get requirements."""
    try:
        with open(REQUIREMENTS) as rqmts:
            return rqmts.read().splitlines()
    except IOError:
        return 'Failed to read {}'.format(REQUIREMENTS)


def long_description():
    """Insert README.md into the package."""
    try:
        with open(README) as readme_fd:
            return readme_fd.read()
    except IOError:
        return 'Failed to read {}'.format(README)


setup(
    author=__authors__,
    author_email=['axel.bellec@outlook.fr', 'maxhalford25@gmail.com'],
    dependency_links=[],
    description='Factorial analysis in Python',
    install_requires=get_requirements(),
    license='MIT',
    long_description=long_description(),
    name=__project__,
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/Prince',
    version=__version__,
)
