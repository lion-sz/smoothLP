# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SmoothLP"
copyright = "2025, Lion Szlagowski"
author = "Lion Szlagowski"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Options for the standalone algorithm PDF -------------------------------

# Keep the HTML documentation rooted at index.rst, while using the detailed
# SLP algorithm page as the root of a short, independently shareable PDF.
latex_documents = [
    (
        "slp_algorithm",
        "smoothlp-slp-algorithm.tex",
        "SmoothLP: Smooth Local Projections",
        author,
        "howto",
    ),
]

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
}
