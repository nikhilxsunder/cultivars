Engine reference
================

The layers behind the public API, for contributors. ``_core`` holds the
pure numerics, validators, converters, and defaults; ``_internals`` the
model bases, fits, mixins, priors, records, and solvers the public classes
delegate to. The public modules appear a second time below with each
class's own private members, the ones it defines itself rather than
inherits; inherited private members are on the page of the base that
defines them. Everything on these pages may change between releases
without notice.

Private layers
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: engine/module.rst

   cultivars._core
   cultivars._internals

Public surface, private members
-------------------------------

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:
   :template: surface/module.rst

   cultivars.univariate
   cultivars.multivariate
   cultivars.state_space
   cultivars.bayes
   cultivars.diagnostics
   cultivars.forecast
   cultivars.spectral
