execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/content/*.ipynb

render-notebooks:
	jupyter nbconvert --to markdown docs/content/*.ipynb
	sed -e '/<script/,/<\/script>/{/^$/d;}' docs/content/pca.md
