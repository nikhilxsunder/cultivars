.. meta::
   :description: The design every cultivars estimator shares: models hold a specification, fit returns a frozen result record, summaries state their assumptions, and errors say what to fix.

Core concepts
=============

Every estimator in cultivars follows the same few conventions. This page
states them once, so that the API reference can assume them.

Model, fit, result
------------------

A model object holds the data and the specification and nothing else.
``fit()`` estimates it and returns a *result*:

.. testcode::

   import numpy as np
   from cultivars.univariate.box_jenkins import ARIMA

   y = np.random.default_rng(0).standard_normal(200).cumsum()
   model = ARIMA(y, order=(1, 1, 0))   # specification: data, orders, trend
   result = model.fit()                # estimation
   print(type(result).__name__)

.. testoutput::

   ARMAResult

Arguments that describe *what model* go to the constructor: orders,
trend, priors, variable names. Arguments that describe *how to
estimate* it go to ``fit()``: the number of posterior draws, a random
seed. One specification can therefore be fitted several ways without
being rebuilt.

Results are records
-------------------

A result is a frozen dataclass. It carries the estimates, the data
needed to forecast and simulate from them, and the diagnostics, all as
attributes, and it cannot be changed after it is built:

.. testcode::

   print(round(result.sigma2, 4), result.nobs, result.is_stationary)
   try:
       result.sigma2 = 1.0
   except AttributeError as error:
       print(type(error).__name__)

.. testoutput::

   0.927 199 True
   FrozenInstanceError

Because results are values rather than handles to a model, any consumer
that reads a result's attributes can take any result that has them:

- a structural identification takes any closed reduced-form result
  (a VAR, a VARMA, an error-correction model);
- a backtest takes any fitting function;
- a model comparison takes any results fitted to the same sample.

None of them needs to know which estimator produced its input. To
change a fit, build a new model and fit it again; the old result stays
exactly as it was.

Summaries state their assumptions
---------------------------------

``summary()`` returns a :class:`~cultivars.summary.SummaryTable`, a
record with four parts:

- ``title``: the specification;
- ``metadata``: label/value pairs such as sample size, log-likelihood
  and information criteria;
- ``rows``: one table of estimates or statistics;
- ``notes``: what the numbers assume and anything the estimator could
  not deliver.

Printing a result prints its summary, and a result left as the last
line of a notebook cell renders it as HTML. The notes are part of the
output, not decoration. A summary that says the AR polynomial has a root
near the unit circle, or that standard errors are unavailable, is
telling you how far to trust the numbers above it.

.. testcode::

   table = result.summary()
   print(table.title)
   print(table.notes[-1])

.. testoutput::

   ARIMA(1, 1, 0) Results
   Standard errors are not yet available for this estimator.

With the ``pandas`` or ``polars`` extra installed, ``to_pandas()`` and
``to_polars()`` turn a summary table into a dataframe, and do the same
for results that carry per-observation output (fitted values and
residuals).

Tests return verdicts
---------------------

A hypothesis test returns a record too, with the same core fields
whatever the test:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``statistic``
     - The test statistic.
   * - ``pvalue``
     - Its p-value under the null.
   * - ``null``
     - The null hypothesis, in words.
   * - ``reject(alpha=0.05)``
     - The verdict at level ``alpha``.
   * - ``summary()``
     - The full table: null, statistic, p-value, critical values where
       they exist, and notes on the reference distribution.

A test run at several settings at once, such as the BDS test at several
embedding dimensions, returns the headline setting and lists the others
in ``companions``.

Structural analysis comes in two stages
---------------------------------------

A reduced form is estimable from the data alone. A structural
interpretation is not: it rests on identifying restrictions that the
data cannot check. cultivars keeps the two apart. A VAR result knows
nothing about economic shocks. A scheme from
:mod:`cultivars.multivariate.structural` takes the fitted reduced form
and returns a separate structural result:

.. code-block:: python

   var = VAR(y, order=4, names=names).fit()                      # reduced form
   svar = RecursiveSVAR(var, order=names).identify()             # point-identified
   accepted = SignRestrictedSVAR(var, signs, draws=1000).identify()  # set-identified

A point-identified scheme returns one structural result. A
set-identified scheme returns the set of accepted models, summarized by
quantiles rather than a point estimate. Either way, the reduced-form
result is untouched, so several schemes can be applied to one fit and
compared directly.

Data in
-------

- **Shape.** A single series is any 1-D array-like of length ``n``; a
  system is an ``(n, k)`` array-like with one column per variable. Rows
  are time periods, oldest first. Lists, NumPy arrays, and pandas or
  polars objects are all accepted and converted to ``float64``.
- **Names.** Pass labels explicitly with ``names=``. Column names on a
  dataframe are not read, and without ``names`` variables are labelled
  ``y1``, ``y2``, and so on. Methods that take a variable, such as
  ``granger_causality("rate", "growth")``, accept these names.
- **Dates.** Estimation works on positions, not dates. To attach dates
  to per-observation output, pass them when exporting, for example
  ``result.to_pandas(index=dates)``.
- **Missing values.** Non-finite values are refused with a
  :class:`~cultivars.exceptions.NumericalError` rather than dropped or
  filled silently. Handle gaps before fitting, so that the choice is
  yours and visible.

Randomness
----------

Every routine that draws random numbers takes a ``seed`` argument
(an integer, a :class:`numpy.random.Generator`, or ``None``), and the
same seed reproduces the same draws. This covers posterior sampling,
rotation draws in sign restrictions, bootstraps and simulation. Fix the
seed in anything you intend to publish.

When something is wrong
-----------------------

Every error cultivars raises on purpose derives from
:class:`~cultivars.exceptions.CultivarsError`. The class tells you what
to fix, and the message names the argument and the values it accepts:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Exception
     - Also a
     - Raised when
   * - :class:`~cultivars.exceptions.DimensionError`
     - ``ValueError``
     - An input has the wrong shape, rank or length.
   * - :class:`~cultivars.exceptions.SpecificationError`
     - ``ValueError``
     - An option is outside its allowed values, an order is too small,
       the sample is too short for what was asked, or a set of
       restrictions cannot be satisfied.
   * - :class:`~cultivars.exceptions.NumericalError`
     - ``RuntimeError``
     - The arithmetic fails: non-finite data, a covariance that is not
       positive semidefinite, a singular system.

.. testcode::

   from cultivars.exceptions import CultivarsError

   try:
       ARIMA(y, order=(1, 1))
   except CultivarsError as error:
       print(type(error).__name__, "-", error)

.. testoutput::

   SpecificationError - order must have 3 elements ('p', 'd', 'q'); got (1, 1).

``except CultivarsError`` catches every problem the library detects and
nothing else. Anything that escapes it, such as an ``AssertionError``
or a raw NumPy error, is a bug, and worth
`reporting <https://github.com/nikhilxsunder/cultivars/issues>`_.

Imports and stability
---------------------

Names are imported from the leaf module that defines them:

.. code-block:: python

   from cultivars.univariate.box_jenkins import ARIMA
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

The subpackages and the top-level ``cultivars`` package re-export
nothing, so an import path always tells you the model family. Names
beginning with an underscore, and everything under
:mod:`cultivars.engine`, are implementation detail and may change
without notice. Until ``1.0.0b1``, public argument names, result fields
and summary wording may still change between pre-releases; each change
is listed in the :doc:`release notes <../project/release_notes>`.
