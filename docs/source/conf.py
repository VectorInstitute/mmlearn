"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("./_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mmlearn"
copyright = "2024, Vector AI Engineering"  # noqa: A001
author = "Vector AI Engineering"
release = "0.1.0a1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.apidoc",
]
add_module_names = False
autoclass_content = "class"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__call__,__getitem__,__iter__,__len__",
    "inherited-members": False,
    "ignore-module-all": True,
}
autodoc_inherit_docstrings = True
autosectionlabel_prefix_document = True
autosummary_generate = True
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True
set_type_checking_flag = True


intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "torchaudio": ("https://pytorch.org/audio/stable/", None),
    "hydra-zen": ("https://mit-ll-responsible-ai.github.io/hydra-zen/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "torchmetrics": ("https://lightning.ai/docs/torchmetrics/stable/", None),
    "Pillow": ("https://pillow.readthedocs.io/en/latest/", None),
    "transformers": ("https://huggingface.co/docs/transformers/en/", None),
    "peft": ("https://huggingface.co/docs/peft/en/", None),
}

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_js_files = ["require.min.js", "custom.js"]
html_additional_pages = {"page": "page.html"}
html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#faad1a",
        "color-brand-content": "#eb088a",
        "color-foreground-secondary": "#52c7de",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/VectorInstitute/mmlearn",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
