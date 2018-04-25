# Contributing

## Development setup

Please [install Anaconda](Anaconda) and [create a virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/), this way it makes it easier to reproduce errors and whatnot.

```sh
>>> conda create -n prince python=3.6 anaconda
>>> source activate prince
```

Then install the necessary dependencies.

```sh
>>> pip install -r requirements.txt
>>> pip install -r requirements.dev.txt
```

## Upload to PyPI

```sh
pip install --update setuptools twine
```

Create `$HOME/.pypirc` with the following content:

```sh
[distutils]
index-servers=
  pypi
  pypitest

[pypi]
repository=https://pypi.org
username=your_username
password=your_password

[pypitest]
repository=https://test.pypi.org
username=your_username
password=your_password
```

```sh
python setup.py sdist upload -r pypitest
python setup.py sdist upload -r pypi
```
