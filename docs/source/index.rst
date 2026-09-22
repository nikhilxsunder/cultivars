cultivars
=========

Research-grade time series econometrics for Python 3.12+: ARIMA to state
space, VAR to structural identification, Bayesian and spectral methods,
on numpy and scipy alone.

.. warning::

   ``cultivars`` |release| is a **prerelease**. The public API is stable
   in shape but not yet frozen; expect argument and result-field renames
   before 1.0.0. Pin the exact version if you depend on it.

Install
-------

.. code-block:: bash

   pip install --pre cultivars

``pandas`` and ``polars`` are optional; ``pip install --pre "cultivars[pandas]"``
enables ``result.to_pandas()`` and dataframe input.

Quick start
-----------

.. code-block:: python

   import numpy as np
   from cultivars.univariate import ARIMA
   from cultivars.multivariate import VAR, RecursiveSVAR

   rng = np.random.default_rng(0)
   y = np.cumsum(rng.standard_normal(300))

   res = ARIMA(y, order=(1, 1, 1)).fit()
   print(res.summary())

   panel = rng.standard_normal((300, 3))
   var = VAR(panel, order=2, names=["y1", "y2", "y3"]).fit()
   svar = RecursiveSVAR(var, order=["y1", "y2", "y3"]).identify()
   irf = svar.irf(horizon=20)          # (horizon + 1, variable, shock)

Every model follows the same pattern: construct with the data and the
specification, call ``fit()`` (or ``identify()``, ``filter()``,
``compute()``, ``date()`` for the non-likelihood tools), and read the
result object -- ``print(res)`` renders a summary table, ``res.to_pandas()``
a frame.

.. toctree::
   :maxdepth: 2
   :hidden:

   api/index
