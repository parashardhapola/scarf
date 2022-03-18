import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Scarf"
copyright = "2021, Parashar Dhapola, Göran Karlsson"
author = ""

extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_external_toc",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "myst_nb",
]

templates_path = ["_templates"]
source_suffix = [".rst"]
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "vignettes/dev",
]
pygments_style = "sphinx"
language = "en"

external_toc_path = "toctree.yml"
external_toc_exclude_missing = False
myst_enable_extensions = [
    "colon_fence",
]

html_css_files = ["styles.css"]
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_favicon = "favicon.ico"
html_logo = "logo.png"
html_title = "Scarf documentation"
html_theme_options = {
    "repository_url": "https://github.com/parashardhapola/scarf",
    "use_repository_button": True,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "path_to_docs": "docs/source",
    "use_download_button": True,
    "use_fullscreen_button": True,
    "single_page": False,
    "home_page_in_toc": True,
    "extra_navbar": "",
    "logo_only": True,
    "show_navbar_depth": 2,
    "toc_title": "Sections",
}

htmlhelp_basename = "ScarfDoc"
man_pages = [(master_doc, "scarf", "Scarf Documentation", [author], 1)]

execution_allow_errors = True
jupyter_execute_notebooks = "auto"
execution_timeout = 5000

import matplotlib

matplotlib.use("agg")
