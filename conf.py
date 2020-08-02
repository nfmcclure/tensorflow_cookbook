# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath("../../tensorflowml/"))

project = 'tensorflow'
copyright = '- Wei MEI (Nick Cafferry).'
author = 'Wei MEI'
author = 'Nick McClure'

version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc', 
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

autoclass_content = 'both'
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = 'Chinese'
html_search_language = 'Chinese'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'remain']
pygments_style = 'default'

html_static_path = ['assets']
html_theme = 'sphinx_rtd_theme'
html_favicon = 'GCC.png'
html_logo = 'GCC.png'
html_theme_options = {
    'logo_only': False,
    'style_nav_header_background': '#343131',
}
