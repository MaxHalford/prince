"""Regression coverage for notebook HTML passing through nbconvert and Hugo."""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import nbformat
import pytest
from nbconvert import MarkdownExporter

DOCS = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("hugo") is None, reason="The docs rendering test requires Hugo")
def test_rich_html_survives_markdown_rendering(tmp_path):
    # Blank lines followed by indented HTML previously became code blocks.
    html = """<div class="estimator">

    <div class="estimator-table">
        <details><summary>Parameters</summary>
        <table><tr><td>n_components</td><td>2</td></tr></table>
        </details>
    </div>
</div>
<script>
const message = `first line

    last line`;
</script>"""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell('+++\ntitle = "Rendering test"\n+++'),
            nbformat.v4.new_code_cell(
                "estimator",
                outputs=[nbformat.v4.new_output("display_data", data={"text/html": html})],
            ),
            nbformat.v4.new_markdown_cell("## After the output"),
        ]
    )
    exporter = MarkdownExporter(template_file=str(DOCS / "templates/hugo.md.j2"))
    markdown, _ = exporter.from_notebook_node(notebook)
    (tmp_path / "hugo.toml").write_text('baseURL = "https://example.com/"\n')
    (tmp_path / "content").mkdir()
    (tmp_path / "content/test.md").write_text(markdown)
    shutil.copytree(DOCS / "layouts/shortcodes", tmp_path / "layouts/shortcodes")
    (tmp_path / "layouts/_default").mkdir()
    (tmp_path / "layouts/_default/single.html").write_text("{{ .Content }}")
    subprocess.run(["hugo", "--source", str(tmp_path)], check=True, capture_output=True)
    rendered = (tmp_path / "public/test/index.html").read_text()
    assert html in rendered
    assert '<div class="notebook-output">' in rendered
    assert '<h2 id="after-the-output">After the output</h2>' in rendered
    assert "&lt;div" not in rendered
