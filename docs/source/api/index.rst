API reference
=============

Every public name lives in one of the modules below, reached through its
subpackage; the subpackages and the top-level ``cultivars`` namespace hold
modules only and re-export nothing. Names beginning with an underscore, and
the ``_core`` and ``_internals`` packages, are implementation detail and may
change between releases without notice.

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   cultivars.univariate
   cultivars.multivariate
   cultivars.state_space
   cultivars.bayes
   cultivars.diagnostics
   cultivars.forecast
   cultivars.spectral
   cultivars.exceptions
   cultivars.typing
