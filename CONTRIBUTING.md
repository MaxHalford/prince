# Contributing

## Setup

```sh
git clone https://github.com/MaxHalford/prince
cd prince
uv sync --extra dev
```

Install [prek](https://github.com/j178/prek) to run code quality checks as git hooks:

```sh
uv tool install prek
prek install --hook-type pre-push
```

You can optionally run the hooks at any time:

```sh
prek run --all-files
```

You can also type check with [ty](https://github.com/astral-sh/ty):

```sh
uvx ty check
```

## Unit tests

Some unit tests call the FactoMineR package via rpy2; you have to install it:

```sh
Rscript -e 'install.packages("FactoMineR", repos="https://cloud.r-project.org")'
```

```sh
uv run pytest
```

## Building docs locally

```sh
make execute-notebooks
make render-notebooks
(cd docs && hugo serve)
```

## Deploy docs

Run the [docs workflow](https://github.com/MaxHalford/prince/actions/workflows/hugo.yml) via the GitHub interface.

## Release

Run the [publishing workflow](https://github.com/MaxHalford/prince/actions/workflows/publish.yml) via the GitHub interface.
