Engine reference
================

The layers behind the public API, for contributors. ``_core`` holds the
pure numerics and validators, ``_internals`` the model bases, fits,
mixins, priors, and solvers the public classes delegate to. The public
subpackages appear a second time below, walked by defining module rather
than by re-export, with each class's own private members alongside its
public ones. Everything on these pages may change between releases
without notice.

Private layers
--------------

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:
   :template: engine/module.rst

   cultivars.engine._core
   cultivars.engine._internals

Public surface, by defining module
----------------------------------

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
