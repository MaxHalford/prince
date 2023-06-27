# Contributing

## Setup

```sh
git clone https://github.com/MaxHalford/prince
cd prince
poetry install
poetry shell
```

Install the [pre-commit](https://pre-commit.com/) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
pre-commit install --hook-type pre-push
```

You can optionally run `pre-commit` at any time as so:

```sh
pre-commit run --all-files
```

## Unit tests

Some unit tests call the FactoMineR package via rpy2; you have to install it:

```sh
Rscript -e 'install.packages("FactoMineR", repos="https://cloud.r-project.org")'
```

```sh
pytest
```

## Building docs locally

```sh
make execute-notebooks
make render-notebooks
(cd docs && hugo serve)
```

## Deploy docs

```sh
gh workflow run hugo.yml
```

## Release

```sh
poetry publish --build
```
