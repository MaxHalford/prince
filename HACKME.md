# Contributing

## To do

- Finish refactoring tests
- Test Matplotlib

## Development setup

Please [install Anaconda](Anaconda) and [create a virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/), this way it makes it easier to reproduce errors and whatnot.

```sh
>>> conda create -n prince-venv python=3.6 anaconda
>>> source activate prince-venv
```

Then install the necessary dependencies.

```sh
>>> pip install -r requirements.dev.txt
```

## Upload to PyPI

[Reference](http://peterdowns.com/posts/first-time-with-pypi.html)

Create `~/.pypirc` with the following content:

```sh
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
repository=https://pypi.python.org/pypi
username=your_username
password=your_password

[pypitest]
repository=https://testpypi.python.org/pypi
username=your_username
password=your_password
```

```sh
python setup.py sdist upload -r pypitest
python setup.py sdist upload -r pypi
```

