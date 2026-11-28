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
from importlib.metadata import version as _distribution_version
from pathlib import Path
from types import FunctionType, MethodType, ModuleType
from typing import TypeAliasType

import sphinx
from sphinx.application import Sphinx
from sphinx.pycode import ModuleAnalyzer
from sphinx.util import logging as sphinx_logging

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
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "myst_parser",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "sphinx_last_updated_by_git",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
    "sphinx_tippy",
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # sphinx-gallery writes notebooks and backreference stubs next to the
    # rendered pages; they are outputs, not sources.
    "auto_examples/**/*.ipynb",
    "auto_examples/**/*.py",
    "gen_modules/backreferences/*.examples",
]

source_suffix: dict[str, str] = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Warnings never fail a build here; the workflow adds ``-W`` only when run
# with strict=true. Private-base cross-refs (``_VectorInferenceMixin``) are
# intentionally undocumented, so nitpicky stays off.
nitpicky: bool = False
suppress_warnings: list[str] = [
    # Inherited dataclass fields whose annotation names a private type that is
    # imported in the base module but not in the subclass module; the rendered
    # type falls back to the literal string, which is what we want anyway.
    "sphinx_autodoc_typehints.forward_reference",
    # Relative Markdown links inside the included root CONTRIBUTING/SECURITY.
    "myst.xref_missing",
    # Unpicklable config values disable the environment cache but not the
    # build; every value below is picklable, this is insurance.
    "config.cache",
]

# sphinx -b doctest runs only explicit .. testcode:: / .. doctest:: blocks.
# Bare >>> blocks in docstrings share one namespace per document under
# this builder and fail by the hundreds; those are pytest --doctest-modules'
# job, where each docstring gets its own globals.
doctest_test_doctest_blocks: str = ""

# -- MyST --------------------------------------------------------------------
myst_enable_extensions: list[str] = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",  # requires linkify-it-py in the docs dependency group
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors: int = 4

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate: bool = True
# Subpackage ``__init__`` files re-export modules only, so honouring their
# ``__all__`` documents each model under its module path
# (cultivars.multivariate.reduced_form.vector_autoregression.VAR).
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
# The engine reference is built and published from the cultivars-engine
# repository, one directory per release. This site links to the directory of
# the release that is installed while it builds, which is the release that
# pyproject.toml pins: the header link and the cross-references into the engine
# then describe the code this version of cultivars actually runs on, and keep
# doing so when an older tag is rebuilt. That release's reference has to be
# published before this site is built, or the link and the inventory are
# missing. DOCS_ENGINE_URL overrides the address, for a preview or for a build
# against an engine release that has no published reference yet.
_ENGINE_DOCS_URL: str = os.environ.get(
    "DOCS_ENGINE_URL",
    f"https://nikhilxsunder.github.io/cultivars-engine/{_distribution_version('cultivars-engine')}/",
)
if not _ENGINE_DOCS_URL.endswith("/"):
    _ENGINE_DOCS_URL += "/"

intersphinx_mapping: dict[str, tuple[str, str | None]] = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "fedfred": ("https://nikhilxsunder.github.io/fedfred/", None),
    "ns-sdn": ("https://nikhilxsunder.github.io/ns-sdn/stable/", None),
    "cultivars-engine": (_ENGINE_DOCS_URL, None),
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
# Origin and path prefix the site is served from, with the live site as the
# default. The deploy workflow passes its DOCS_BASE_URL; ``make preview``
# passes http://localhost:<port>/ and mirrors the live layout (switcher.json
# at the root, one directory per version), so the same switcher works in
# both. Canonical-URL metadata (html_baseurl, opengraph, sitemap) stays on
# the live stable URL regardless.
_DOCS_SITE_URL: str = os.environ.get("DOCS_SITE_URL", "https://nikhilxsunder.github.io/cultivars/")
if not _DOCS_SITE_URL.endswith("/"):
    _DOCS_SITE_URL += "/"

# Every build declares stable as canonical: dev and archives consolidate to it.
html_baseurl: str = "https://nikhilxsunder.github.io/cultivars/stable/"

# Only stable publishes a sitemap; nothing else is meant to be indexed.
if _DOCS_VERSION != "stable":
    extensions.remove("sphinx_sitemap")

# -- HTML output -------------------------------------------------------------
html_last_updated_fmt: str = "%Y-%m-%d"

html_theme: str = "pydata_sphinx_theme"
html_title: str = "cultivars"
html_logo: str = "_static/cultivars-logo.png"
html_favicon: str = "_static/cultivars-favicon.ico"
html_static_path: list[str] = ["_static"]
# robots.txt and the IndexNow key live at the site root and are written by
# the deploy workflow's ``root`` job from docs/root/ and a secret; per-version
# copies under stable/ are ignored by crawlers and would duplicate the key.
html_extra_path: list[str] = []
html_css_files: list[str] = ["custom.css", "consent.css"]
html_js_files: list[tuple[str, dict[str, str]]] = [("consent.js", {"defer": "defer"})]
html_show_sourcelink: bool = False

html_theme_options: dict[str, object] = {
    "analytics": {"google_analytics_id": "G-QPCZ8H078M"},
    "logo": {
        "image_light": "_static/cultivars-logo.png",
        "image_dark": "_static/cultivars-logo-dark.png",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": _DOCS_SITE_URL + "switcher.json",
        "version_match": _DOCS_VERSION,
    },
    "check_switcher": False,
    "show_version_warning_banner": False,
    "navbar_center": ["navbar-dropdowns"],
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
        # Re-enable with real URLs once the badges exist; an empty url renders
        # a self-link and warns in current pydata-sphinx-theme.
        # {"name": "OpenSSF", "url": "", "icon": "fas fa-trophy"},
        # {"name": "Zenodo", "url": "", "icon": "fas fa-book"},
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "footer_start": ["cultivars-footer"],
    "footer_center": [],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

# Repository files that are linked from the header menu but not built into the
# site, at the branch or tag this build documents.
_GITHUB_BLOB: str = f"https://github.com/nikhilxsunder/cultivars/blob/{_DOCS_GIT_REF}"

html_context: dict[str, object] = {
    "github_user": "nikhilxsunder",
    "github_repo": "cultivars",
    "doc_path": "docs/source",
    "github_version": _DOCS_GIT_REF,
    # Header menus, read by _templates/navbar-dropdowns.html. ``doc`` is the
    # page a section title links to; ``children`` are (target, label) pairs
    # shown in its dropdown, and an empty list renders a plain link. ``doc``
    # and each target are a docname relative to this directory, or an absolute
    # URL for a page that lives outside the site. This list is maintained by hand: it
    # does not follow the toctree, so a new section, guide, example or
    # subpackage has to be added here as well.
    "navbar_dropdowns": [
        {
            "title": "Getting started",
            "doc": "getting_started/index",
            "children": [
                ("getting_started/installation", "Installation"),
                ("getting_started/quickstart", "Quickstart"),
                ("getting_started/concepts", "Core concepts"),
            ],
        },
        {
            "title": "User guide",
            "doc": "user_guide/index",
            "children": [
                ("user_guide/choosing_a_model", "Choosing a model"),
                ("user_guide/diagnostics", "Diagnostics and testing"),
                ("user_guide/univariate_linear", "Univariate linear models"),
                ("user_guide/volatility", "Volatility models"),
                ("user_guide/var", "Vector autoregressions"),
                ("user_guide/structural_identification", "Structural identification"),
                ("user_guide/bayesian_vars", "Bayesian VARs"),
                ("user_guide/state_space", "State-space models"),
                ("user_guide/spectral", "Spectral analysis"),
                ("user_guide/forecasting", "Forecasting and evaluation"),
            ],
        },
        {
            "title": "Examples",
            "doc": "auto_examples/index",
            "children": [
                ("auto_examples/plot_01_monetary_policy_var", "Monetary policy VAR"),
                ("auto_examples/plot_02_blanchard_quah", "Long-run restrictions"),
                (
                    "auto_examples/plot_03_oil_market_sign_restrictions",
                    "Oil market sign restrictions",
                ),
                (
                    "auto_examples/plot_04_heteroskedasticity_and_proxy",
                    "Heteroskedasticity and proxies",
                ),
                ("auto_examples/plot_05_nelson_siegel_yield_curve", "Nelson-Siegel yield curve"),
                ("auto_examples/plot_06_business_cycle_regimes", "Business-cycle regimes"),
                ("auto_examples/plot_07_great_ratios_vecm", "Cointegration and a VECM"),
                ("auto_examples/plot_08_large_bvar_forecasting", "Bayesian VAR forecasting"),
                (
                    "auto_examples/plot_09_volatility_models_var_backtest",
                    "Equity volatility models",
                ),
                (
                    "auto_examples/plot_10_inflation_forecast_evaluation",
                    "Inflation forecast evaluation",
                ),
            ],
        },
        {
            "title": "API reference",
            "doc": "api/index",
            "children": [
                ("api/generated/cultivars.univariate", "cultivars.univariate"),
                ("api/generated/cultivars.multivariate", "cultivars.multivariate"),
                ("api/generated/cultivars.state_space", "cultivars.state_space"),
                ("api/generated/cultivars.diagnostics", "cultivars.diagnostics"),
                ("api/generated/cultivars.forecast", "cultivars.forecast"),
                ("api/generated/cultivars.spectral", "cultivars.spectral"),
                ("api/generated/cultivars.bayes", "cultivars.bayes"),
                ("api/generated/cultivars.data", "cultivars.data"),
                ("api/generated/cultivars.experimental", "cultivars.experimental"),
            ],
        },
        {"title": "Engine reference", "doc": _ENGINE_DOCS_URL, "children": []},
        {
            "title": "Project",
            "doc": "project/index",
            "children": [
                ("project/release_notes", "Release notes"),
                (f"{_GITHUB_BLOB}/ROADMAP.md", "Roadmap"),
                (f"{_GITHUB_BLOB}/CONTRIBUTING.md", "Contributing"),
                (f"{_GITHUB_BLOB}/SECURITY.md", "Security policy"),
                (f"{_GITHUB_BLOB}/CODE_OF_CONDUCT.md", "Code of conduct"),
                ("https://github.com/nikhilxsunder/cultivars/issues", "Issue tracker"),
            ],
        },
    ],
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

# -- bibtex --
bibtex_bibfiles: list[str] = ["references.bib"]
bibtex_default_style: str = "unsrt"
bibtex_reference_style: str = "author_year"

# -- plot_directive --
plot_include_source: bool = True
plot_html_show_source_link: bool = False
plot_html_show_formats: bool = False
plot_formats: list[tuple[str, int]] = [("png", 150)]
plot_pre_code: str = "import numpy as np\nimport matplotlib.pyplot as plt\nimport cultivars"

# -- sphinx-gallery ----------------------------------------------------------
# Every value is picklable (sort keys by name, not by class) so Sphinx can
# cache the environment between builds.
sphinx_gallery_conf: dict[str, object] = {
    "examples_dirs": [str(_ROOT / "examples")],  # needs examples/GALLERY_HEADER.rst
    "gallery_dirs": ["auto_examples"],  # relative to docs/source
    "filename_pattern": r"/plot_\d{2}_\w+\.py$",  # plot_NN_name.py only; rejects "name 2.py"
    "ignore_pattern": r"/_",
    "within_subsection_order": "ExampleTitleSortKey",
    "doc_module": ("cultivars",),
    "reference_url": {"cultivars": None},  # None = link into this build
    "backreferences_dir": "gen_modules/backreferences",
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("matplotlib",),
    "capture_repr": ("_repr_html_", "__repr__"),
    "matplotlib_animations": False,
    "remove_config_comments": True,
    "download_all_examples": False,
    "show_memory": False,
    "show_signature": False,
    "min_reported_time": 10,
    # A failing example is rendered with its traceback and reported as a
    # Sphinx warning; it fails the build only under -W (workflow strict=true).
    "abort_on_example_error": False,
    "only_warn_on_example_error": True,
    "plot_gallery": os.environ.get("DOCS_PLOT_GALLERY", "1") == "1",
    "default_thumb_file": str(_ROOT / "docs" / "source" / "_static" / "cultivars-logo.png"),
}

# Keep the generated reference out of the sidebar below a fixed depth.
#
# pydata-sphinx-theme writes the whole toctree of a section into every page of
# it, so a section of N pages costs N links on each of N pages. With a page per
# member that was 4,067 links and 775 KB of sidebar on every engine page and
# 4.5 GB for one version of the site, against the 10 GB that GitHub Pages
# accepts for all versions together. The pages are still built and still
# linked from the tables on their parent page; they are only left out of the
# navigation tree. The API reference keeps modules, classes and functions in
# the sidebar; the engine reference keeps modules.
#
# sphinx-remove-toctrees matches these as filesystem globs over the stub
# files, suffix included. A class is the first dotted segment that starts with
# a capital, and in the engine anything three segments below
# ``cultivars_engine`` sits under a module. ``.github/workflows/docs.yml``
# passes the same list on the command line, so that a release tag cut before
# this change is rebuilt with it; keep the two in step.
remove_from_toctrees: list[str] = [
    "gen_modules/backreferences/*",
    "api/generated/*.[A-Z]*.*.rst",
    "engine/generated/cultivars_engine.*.*.*.rst",
    "engine/generated/*.[A-Z]*.rst",
]

# -- sphinx-tippy ------------------------------------------------------------
tippy_props: dict[str, object] = {
    "placement": "auto-start",
    "maxWidth": 500,
    "theme": "material",
    "interactive": True,
    "delay": [200, 0],
}
tippy_anchor_parent_selector: str = "article.bd-article"  # pydata body only; skips navbar/sidebars
tippy_skip_anchor_classes: list[str] = ["headerlink", "sd-stretched-link"]
tippy_enable_mathjax: bool = True  # tips for equation references
tippy_enable_wikitips: bool = False  # build-time network fetch; low value here
tippy_enable_doitips: bool = False
tippy_add_class: str = "has-tippy"

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


_ENGINE_PACKAGE: str = "cultivars_engine"

_LOGGER = sphinx_logging.getLogger(__name__)


def _document_reexports_as_classes() -> None:
    """Document engine classes under their public names as classes, not aliases.

    The public modules re-export their records, results and tests from the
    engine (``from cultivars_engine._internals._tests import _JohansenRankTest as
    JohansenRankTest``), so the object autodoc imports for
    ``cultivars.diagnostics.cointegration.JohansenRankTest`` carries the
    private ``__name__``. autodoc reads any name mismatch as an alias and
    renders ``alias of _JohansenRankTest`` with no signature, docstring or
    members. That is right for ``Short = LongName`` inside one module and
    wrong here, where the public name is the one a reader is meant to see.

    autodoc exposes no event for that decision, so the two properties that
    carry it are wrapped. ``doc_as_attr`` answers ``False`` for a top-level
    class defined under ``cultivars_engine`` and documented outside it.
    ``canonical_full_name`` answers ``None`` for the same objects: the engine
    reference documents the private name in its own right, and a
    ``:canonical:`` alias from each public home would collide when a class
    has two of them. The classes themselves are not touched, so attribute
    docstrings and source links keep resolving through the defining module.

    ``_ClassDefProperties`` is private to Sphinx and identical in 9.1.1 and
    9.2.0, the two versions the lockfile resolves. If a later release moves
    it, a warning is logged and the pages fall back to ``alias of``; the
    strict build turns that warning into a failure.
    """
    try:
        from sphinx.ext.autodoc._property_types import _ClassDefProperties

        inherited_alias = _ClassDefProperties.doc_as_attr.fget
        inherited_canonical = _ClassDefProperties.canonical_full_name.fget
    except (ImportError, AttributeError):
        inherited_alias = inherited_canonical = None
    if inherited_alias is None or inherited_canonical is None:
        _LOGGER.warning(
            "autodoc internals moved in Sphinx %s: engine classes re-exported by the "
            "public modules will render as 'alias of ...'. Update "
            "_document_reexports_as_classes in conf.py.",
            sphinx.__version__,
        )
        return

    def is_reexport(props: _ClassDefProperties) -> bool:
        defined_in = props._obj___module__ or ""
        return (
            len(props.parts) == 1
            and props._obj___name__ is not None
            and defined_in.startswith(f"{_ENGINE_PACKAGE}.")
            and props.module_name != _ENGINE_PACKAGE
            and not props.module_name.startswith(f"{_ENGINE_PACKAGE}.")
        )

    def doc_as_attr(props: _ClassDefProperties) -> bool:
        return False if is_reexport(props) else inherited_alias(props)

    def canonical_full_name(props: _ClassDefProperties) -> str | None:
        return None if is_reexport(props) else inherited_canonical(props)

    _ClassDefProperties.doc_as_attr = property(doc_as_attr)
    _ClassDefProperties.canonical_full_name = property(canonical_full_name)


def _skip_foreign_modules(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    skip: bool,
    options: object,
) -> bool | None:
    """Keep third-party modules re-exported by the public surface out of the reference.

    ``cultivars.data`` re-exports ``fedfred`` as ``economic`` and lists it in
    ``__all__``. autosummary would otherwise recurse into it as a submodule
    and write a stub for every fedfred object under a module path
    (``economic.FredAPI``) that autodoc cannot import, which is one failed
    page per object. Those packages are documented at their own sites, as the
    ``cultivars.data`` docstring says.
    """
    if isinstance(obj, ModuleType) and obj.__name__.partition(".")[0] != project:
        return True
    return None


def setup(app: Sphinx) -> None:
    """Setup hook for Sphinx."""
    _document_reexports_as_classes()
    app.connect("autodoc-process-docstring", _document_type_alias)
    app.connect("autodoc-skip-member", _skip_foreign_modules)
