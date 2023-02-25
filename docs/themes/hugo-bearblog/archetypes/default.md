+++
title = "{{ replace .Name "-" " " | title }}"
date = "{{ .Date }}"

#
# Set menu to "main" to add this page to
# the main menu on top of the page
#
menu = "main"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

#
# tags are optional
#
# tags = [{{ range $plural, $terms := .Site.Taxonomies }}{{ range $term, $val := $terms }}"{{ printf "%s" $term }}",{{ end }}{{ end }}]
+++

This is a page about »{{ replace .Name "-" " " | title }}«.
