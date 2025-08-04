execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace docs/content/*.ipynb

render-notebooks:
	uv run jupyter nbconvert --to markdown docs/content/*.ipynb
