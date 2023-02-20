jupyter nbconvert --to markdown docs/content/*.ipynb

# Remove empty lines in JavaScript outputs
# This is needed for the MarkDown to be parsed correctly by Hugo
for f in docs/content/*.md; do
    sed -e '/<script/,/<\/script>/{/^$/d;}' ${f} > ${f}.tmp
    mv ${f}.tmp ${f}
done
