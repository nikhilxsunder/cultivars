.. meta::
   :description: Install cultivars with pip or uv, choose the pandas or polars extra, pin a pre-release, and set up a development checkout.

Installation
============

Requirements
------------

cultivars supports CPython 3.12, 3.13 and 3.14 on Linux, macOS and
Windows. The numerical core needs only :mod:`numpy` and :mod:`scipy`.
Dataframe export is optional and comes in through an extra:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Extra
     - Installs
     - Switches on
   * - *(none)*
     - numpy, scipy
     - Every model, test, filter and result. Inputs are array-likes;
       outputs are :class:`numpy.ndarray` and plain records.
   * - ``pandas``
     - pandas
     - ``to_pandas()`` on results and summary tables.
   * - ``polars``
     - polars
     - ``to_polars()`` on results and summary tables.

.. note::

   Pre-releases currently also install ``fedfred`` and ``matplotlib``,
   which the :doc:`examples <../auto_examples/index>` use to download
   FRED data and draw figures. Nothing in the library itself imports
   either.

Install a pre-release
---------------------

Every release so far is a pre-release, so installers skip cultivars
unless asked for pre-releases explicitly.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install --pre cultivars
         pip install --pre "cultivars[pandas]"

   .. tab-item:: uv

      .. code-block:: bash

         uv add --prerelease allow cultivars
         uv add --prerelease allow "cultivars[pandas]"

   .. tab-item:: Pinned

      .. parsed-literal::

         pip install "cultivars==\ |release|\ "

      An exact pin needs no ``--pre`` flag, since the version string
      already names the pre-release.

Pin the exact version in anything that must reproduce: a paper's
replication package, a CI job, a lockfile. Between pre-releases,
argument names, result fields and summary wording can still change.
Every such change is listed in the
:doc:`release notes <../project/release_notes>`. The public API freezes
at ``1.0.0b1``. A conda-forge package follows the first stable release.

Check the installation
----------------------

.. code-block:: bash

   python -c "from importlib.metadata import version; print(version('cultivars'))"

The same string is available inside Python as
``cultivars.__about__.__version__``.

Data for the examples
---------------------

The examples download their data from FRED through
`fedfred <https://nikhilxsunder.github.io/fedfred/>`_, which needs a
free API key from the
`St. Louis Fed <https://fred.stlouisfed.org/docs/api/api_key.html>`_.
Set it once in your shell:

.. code-block:: bash

   export FRED_API_KEY=your_key

The quickstart and every docstring example run on simulated data and
need no key.

Development install
-------------------

A development checkout uses `uv <https://docs.astral.sh/uv/>`_ and the
``dev`` dependency group, which brings in the test, lint, type-checking
and documentation tools:

.. code-block:: bash

   git clone https://github.com/nikhilxsunder/cultivars.git
   cd cultivars
   git switch dev
   uv sync --group dev
   uv run pre-commit install

Then:

.. code-block:: bash

   uv run ruff check src/cultivars          # lint
   uv run mypy                              # type-check
   uv run make -C docs html                 # docs, running the examples
   DOCS_PLOT_GALLERY=0 uv run make -C docs html   # docs, skipping the examples

Keep the checkout outside folders a sync service mirrors, such as
iCloud Drive's Desktop & Documents or Dropbox. Those services create
numbered copies (``file 2.py``) of files that tools rewrite quickly,
and Python and Sphinx pick the copies up as real files.

Troubleshooting
---------------

``No matching distribution found for cultivars``
   The installer is skipping pre-releases. Add ``--pre`` (pip),
   ``--prerelease allow`` (uv), or pin the exact version.

``Requires-Python >=3.12``
   cultivars uses syntax from Python 3.12. Create an environment with a
   supported interpreter, for example
   ``uv venv --python 3.13``.

``ImportError: pandas is required for this conversion``
   A ``to_pandas()`` call needs the ``pandas`` extra, and ``to_polars()``
   the ``polars`` extra. Install the one the message names, or read the
   result's array attributes directly.
