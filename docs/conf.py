# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Sphinx documentation builder
"""

# General options:
from pathlib import Path
import sys

try:
    from importlib.metadata import version as metadata_version
except ImportError:
    # Python <3.8 fallback
    from importlib_metadata import version as metadata_version

project = "Quantum Random Access Optimization"
copyright = "2022"  # pylint: disable=redefined-builtin
author = "James R. Garrison et al."

_rootdir = Path(__file__).parent.parent
sys.path.insert(0, str(_rootdir))

# The full version, including alpha/beta/rc tags
release = metadata_version("qrao")
# The short X.Y version
version = version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
]
templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qrao."]

# html theme options
html_static_path = ["_static"]
html_logo = "_static/images/logo.png"

# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "always"
nbsphinx_widgets_path = ""
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "tutorials",
    "how_tos",
    "test_notebooks",
    "background",
]
