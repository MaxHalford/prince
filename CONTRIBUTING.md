# Contributing

## Development setup

Please [install Anaconda](Anaconda) and [create a virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/), this way it makes it easier to reproduce errors and whatnot.

```sh
> conda create -n prince python=3.7 anaconda
> conda activate prince
```

Then install the necessary dependencies.

```sh
> pip install -e ".[dev]"
> python setup.py develop
```

## Upload to PyPI

```sh
> python3 -m pip install --user --upgrade setuptools wheel
> python3 setup.py sdist bdist_wheel
> python3 -m pip install --user --upgrade twine
> python3 -m twine upload dist/*
```
