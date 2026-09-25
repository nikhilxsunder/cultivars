"""Sphinx configuration file for the cultivars project."""
# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import importlib
import inspect
import os
import sys
import tomllib
from pathlib import Path
from types import FunctionType, MethodType, ModuleType
from typing import TypeAliasType

from sphinx.application import Sphinx
from sphinx.pycode import ModuleAnalyzer

# -- Path setup --------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

# -- Project information -----------------------------------------------------
with open(_ROOT / "pyproject.toml", "rb") as _f:
    _META = tomllib.load(_f)["project"]

from cultivars.__about__ import __version__  # noqa: E402

project: str = "cultivars"
copyright: str = "2026, Nikhil Sunder"
author: str = "Nikhil Sunder"
release: str = __version__
version: str = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "myst_parser",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "sphinx_last_updated_by_git",
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = []

source_suffix: dict[str, str] = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Sphinx >= 8 fails the build on missing references only if nitpicky is on.
# Keep it off: private-base cross-refs (``_VectorInferenceMixin``) are
# intentionally undocumented.
nitpicky: bool = False
suppress_warnings: list[str] = [
    # Inherited dataclass fields whose annotation names a private type that is
    # imported in the base module but not in the subclass module; the rendered
    # type falls back to the literal string, which is what we want anyway.
    "sphinx_autodoc_typehints.forward_reference",
    # Relative Markdown links inside the included root CONTRIBUTING/SECURITY.
    "myst.xref_missing",
]

# -- MyST --------------------------------------------------------------------
myst_enable_extensions: list[str] = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors: int = 4

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate: bool = True
# Respect ``__all__`` in each subpackage ``__init__`` so re-exported classes
# are documented under their public path (cultivars.univariate.AR, not
# cultivars.univariate.autoregression.AR).
autosummary_ignore_module_all: bool = False
autosummary_imported_members: bool = False

# Member selection lives in the autosummary templates (``_templates/autosummary``)
# rather than here: Sphinx >= 9 does not reliably honour ``:no-members:`` on a
# directive when ``members`` is a global default, and the module pages must
# not re-document what the class pages already own.
autodoc_default_options: dict[str, str] = {
    "member-order": "groupwise",
    "exclude-members": "__init__, __new__, __weakref__",
}
autodoc_typehints: str = "description"
autodoc_typehints_format: str = "short"
autodoc_typehints_description_target: str = "documented_params"
autodoc_class_signature: str = "mixed"
autodoc_member_order: str = "groupwise"
autodoc_preserve_defaults: bool = True

# sphinx_autodoc_typehints
typehints_fully_qualified: bool = False
always_document_param_types: bool = False
typehints_document_rtype: bool = True
typehints_use_signature: bool = False
typehints_use_signature_return: bool = False
simplify_optional_unions: bool = True

# -- napoleon (Google style, matches ruff pydocstyle convention) --------------
napoleon_google_docstring: bool = True
napoleon_numpy_docstring: bool = False
napoleon_include_init_with_doc: bool = False
napoleon_include_private_with_doc: bool = False
napoleon_use_admonition_for_examples: bool = True
napoleon_use_admonition_for_notes: bool = True
napoleon_use_admonition_for_references: bool = False
napoleon_use_ivar: bool = True  # dataclass fields: avoid duplicate attribute entries
napoleon_use_param: bool = True
napoleon_use_rtype: bool = False
napoleon_preprocess_types: bool = True
napoleon_attr_annotations: bool = True
napoleon_custom_sections: list[tuple[str, str]] = [
    ("Shapes", "params_style"),
    ("Identification", "notes_style"),
    ("Priors", "params_style"),
]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping: dict[str, tuple[str, str | None]] = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "fedfred": ("https://nikhilxsunder.github.io/fedfred/", None),
}
intersphinx_timeout: int = 10

# -- extlinks ----------------------------------------------------------------
extlinks: dict[str, tuple[str, str]] = {
    "github": ("https://github.com/nikhilxsunder/cultivars/%s", "GitHub: %s"),
    "issue": ("https://github.com/nikhilxsunder/cultivars/issues/%s", "issue %s"),
    "doi": ("https://doi.org/%s", "doi:%s"),
    "numpy-doc": ("https://numpy.org/doc/stable/reference/%s", "NumPy Docs: %s"),
    "scipy-doc": ("https://docs.scipy.org/doc/scipy/reference/%s", "SciPy Docs: %s"),
}

# -- switcher ---------------------------------------------------------------

_DOCS_VERSION: str = os.environ.get("DOCS_VERSION", "dev")  # dev | stable | <release>
_DOCS_GIT_REF: str = os.environ.get("DOCS_GIT_REF", "dev")  # branch or tag, for Edit on GitHub

# Every build declares stable as canonical: dev and archives consolidate to it.
html_baseurl: str = "https://nikhilxsunder.github.io/cultivars/stable/"

# Only stable publishes a sitemap; nothing else is meant to be indexed.
if _DOCS_VERSION != "stable":
    extensions.remove("sphinx_sitemap")

# -- HTML output -------------------------------------------------------------
html_theme: str = "pydata_sphinx_theme"
html_title: str = "cultivars"
html_logo: str = "_static/cultivars-logo.png"
html_favicon: str = "_static/cultivars-favicon.ico"
html_static_path: list[str] = ["_static"]
# html_extra_path: list[str] = ["robots.txt"]
html_css_files: list[str] = ["custom.css", "consent.css"]
html_js_files: list[tuple[str, dict[str, str]]] = [("consent.js", {"defer": "defer"})]
# html_js_files: list[str] = ["json_ld.js"]
html_show_sourcelink: bool = False

html_theme_options: dict[str, object] = {
    "analytics": {"google_analytics_id": "G-QPCZ8H078M"},
    "logo": {
        "image_light": "_static/cultivars-logo.png",
        "image_dark": "_static/cultivars-logo.png",
    },
    "header_links_before_dropdown": 3,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": "https://nikhilxsunder.github.io/cultivars/switcher.json",
        "version_match": _DOCS_VERSION,
    },
    "check_switcher": False,
    "show_version_warning_banner": False,
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nikhilxsunder/cultivars",
            "icon": "fab fa-github",
        },
        {"name": "PyPI", "url": "https://pypi.org/project/cultivars/", "icon": "fab fa-python"},
        {
            "name": "Conda-Forge",
            "url": "https://anaconda.org/conda-forge/cultivars",
            "icon": "fas fa-database",
        },
        {
            "name": "Codecov",
            "url": "https://app.codecov.io/gh/nikhilxsunder/cultivars",
            "icon": "fas fa-umbrella",
        },
        {
            "name": "OpenSSF",
            "url": "",
            "icon": "fas fa-trophy",
        },
        {"name": "Zenodo", "url": "", "icon": "fas fa-book"},
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

html_context: dict[str, str] = {
    "github_user": "nikhilxsunder",
    "github_repo": "cultivars",
    "doc_path": "docs/source",
    "github_version": _DOCS_GIT_REF,
}

html_meta: dict[str, str] = {
    "description": _META["description"],
    "keywords": ", ".join(_META["keywords"]),
}

# -- sitemap -----------------------------------------------------------------
sitemap_filename: str = "sitemap.xml"
sitemap_url_scheme: str = "{link}"

# -- opengraph ---------------------------------------------------------------
ogp_site_url: str = html_baseurl
ogp_image: str = html_baseurl + "_static/cultivars-logo.png"
ogp_description_length: int = 300
ogp_type: str = "website"
ogp_enable_meta_description: bool = True
ogp_custom_meta_tags: list[str] = [
    '<meta property="og:locale" content="en_US" />',
    '<meta property="og:site_name" content="cultivars Documentation" />',
    '<meta property="og:image:alt" content="cultivars Logo" />',
]

# -- doctest -----------------------------------------------------------------
doctest_global_setup: str = "import numpy as np\nimport cultivars"


# -- linkcode ----------------------------------------------------------------
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Link each documented object to its source lines on GitHub."""
    if domain != "py" or not info["module"]:
        return None
    obj: object = importlib.import_module(info["module"])
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    if isinstance(obj, property):
        obj = obj.fget
    if isinstance(obj, (FunctionType, MethodType)):
        target: ModuleType | type[object] | FunctionType | MethodType = inspect.unwrap(obj)
    elif isinstance(obj, (ModuleType, type)):
        target = obj
    else:
        return None
    try:
        source_file = inspect.getsourcefile(target)
        lines, first = inspect.getsourcelines(target)
    except (OSError, TypeError):
        return None
    if source_file is None:
        return None
    try:
        relative = Path(source_file).resolve().relative_to(_ROOT)
    except ValueError:
        return None
    last = first + len(lines) - 1
    return (
        f"https://github.com/nikhilxsunder/cultivars/blob/{_DOCS_GIT_REF}"
        f"/{relative.as_posix()}#L{first}-L{last}"
    )


# -- autodoc hooks -----------------------------------------------------------
def _document_type_alias(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: dict[str, object],
    lines: list[str],
) -> None:
    """Document PEP 695 ``type`` aliases re-exported through ``cultivars.typing``.

    A ``TypeAliasType`` carries no ``__doc__`` of its own, so autodoc falls back
    to the generic ``typing.TypeAliasType`` docstring. Replace it with the
    docstring-comment that follows the alias in its defining module, and lead
    with the aliased value so the permitted literals are visible.
    """
    if what != "data" or not isinstance(obj, TypeAliasType):
        return
    value = repr(obj.__value__).replace("typing.", "")
    doc: list[str] = []
    module = obj.__module__
    if module is not None:
        try:
            attr_docs = ModuleAnalyzer.for_module(module).find_attr_docs()
            doc = list(attr_docs.get(("", obj.__name__), []))
        except Exception:  # pragma: no cover - analyzer failure is non-fatal
            doc = []
    lines[:] = [f"Alias of ``{value}``.", "", *doc]


def setup(app: Sphinx) -> None:
    """Setup hook for Sphinx."""
    app.connect("autodoc-process-docstring", _document_type_alias)
