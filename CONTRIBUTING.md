# Contributing

```sh
git clone https://github.com/MaxHalford/prince
cd prince
poetry install
poetry shell
pytest
```

This is how to build and serve the docs locally:

```sh
make execute-notebooks
make render-notebooks
(cd docs && hugo serve)
```
