# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Scarf'
copyright = '2021, Parashar Dhapola, Göran Karlsson'
author = 'Parashar Dhapola, Göran Karlsson'

extensions = [
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'vignettes/dev']

language = None

pygments_style = 'sphinx'

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_favicon = '_static/scarf_icon.svg'
html_logo = '_static/scarf_logo_inversed.svg'
# html_title = project + ' version ' + release
# html_sidebars = {}
html_theme_options = {
    'style_nav_header_background': 'black',
}

htmlhelp_basename = 'ScarfDoc'

latex_elements = {}
latex_documents = [
    (master_doc, 'Scarf.tex', 'Scarf Documentation',
     'Parashar Dhapola', 'manual'),
]
man_pages = [
    (master_doc, 'scarf', 'Scarf Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'Scarf', 'Scarf documentation',
     author, 'Scarf', 'One line description of project.',
     'Miscellaneous'),
]

nbsphinx_custom_formats = {
    '.md': ['jupytext.reads', {'fmt': 'md'}],
}
nbsphinx_execute = 'auto'
nbsphinx_kernel_name = 'python'
nbsphinx_allow_errors = True

import matplotlib
matplotlib.use('agg')
