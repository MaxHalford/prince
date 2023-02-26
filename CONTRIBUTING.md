# Contributing

## Setup

```sh
git clone https://github.com/MaxHalford/prince
cd prince
poetry install
poetry shell
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
