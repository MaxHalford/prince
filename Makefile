execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace docs/content/*.ipynb

render-notebooks:
	uv run jupyter nbconvert --to markdown docs/content/*.ipynb
	(for f in docs/content/*.md; do sed -e '/<script/,/<\/script>/{/^$/d;}' ${f} > ${f}.tmp; mv ${f}.tmp ${f}; done)
