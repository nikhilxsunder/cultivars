# filepath: /src/cultivars/engine/_internals/__init__.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The model layer of the engine: records, substrates, objectives, and estimation bases.

``engine._core`` holds the numerical primitives, which know nothing about
any model. This package holds everything that does: the state-space
substrates the models are written on, the objectives their optimizers
minimize, the samplers, the estimation base classes the public fronts
subclass, and the records that carry results between those layers. Nothing
here is public by name. The public packages subclass or re-export what
they need under public names.

The modules, by role.

Records, frozen dataclasses that carry state between layers:

* :mod:`~cultivars.engine._internals._parameters`: parameter records, the meeting point of a flat
  optimization vector and a model.
* :mod:`~cultivars.engine._internals._fits`: raw output of estimators and samplers.
* :mod:`~cultivars.engine._internals._results`: filter and smoother output, and
  the shared result bases.
* :mod:`~cultivars.engine._internals._posteriors`: packed sampler output with
  its derived quantities.
* :mod:`~cultivars.engine._internals._moments`,
  :mod:`~cultivars.engine._internals._coefficients`,
  :mod:`~cultivars.engine._internals._levels`,
  :mod:`~cultivars.engine._internals._layouts`,
  :mod:`~cultivars.engine._internals._states`,
  :mod:`~cultivars.engine._internals._solutions`: smaller intermediate records.
* :mod:`~cultivars.engine._internals._assessments`,
  :mod:`~cultivars.engine._internals._selections`,
  :mod:`~cultivars.engine._internals._tests`: verdicts (stability, lag order
  and model weights, hypothesis and convergence tests).
* :mod:`~cultivars.engine._internals._covariances`: coefficient uncertainty,
  sampling or posterior, behind the
  :mod:`~cultivars.engine._internals._inferences` protocol.

Interfaces that outside code implements:

* :mod:`~cultivars.engine._internals._priors`: the coefficient-prior interface and the volatility
  hyperpriors.
* :mod:`~cultivars.engine._internals._predictors` and
  :mod:`~cultivars.engine._internals._engines`: the fitted mean map and the
  training backend of the neural mean-function models.
* :mod:`~cultivars.engine._internals._specifications`: what a perturbation model must supply.

Machinery:

* :mod:`~cultivars.engine._internals._substrates`: the state-space models with
  their filters and smoothers.
* :mod:`~cultivars.engine._internals._filters` and
  :mod:`~cultivars.engine._internals._smoothers`: the Hamilton filter and the
  Kim smoother for a discrete regime chain.
* :mod:`~cultivars.engine._internals._systems` and
  :mod:`~cultivars.engine._internals._emitters`: parameter records turned into
  state-space systems.
* :mod:`~cultivars.engine._internals._objectives` and
  :mod:`~cultivars.engine._internals._means`: the functions the optimizers
  minimize and the conditional-mean layers they share.
* :mod:`~cultivars.engine._internals._solvers`: optimizers, posterior updates,
  and the perturbation solver.
* :mod:`~cultivars.engine._internals._simulators`: data generation from fitted or stated parameters.
* :mod:`~cultivars.engine._internals._chains`: particle marginal Metropolis-Hastings.

Bases:

* :mod:`~cultivars.engine._internals._models`: the estimation engines the
  public model classes subclass.
* :mod:`~cultivars.engine._internals._mixins`: behavior shared by result records (summaries, series,
  comparison, inference, convergence, replication).

Every submodule is imported here so that the package is fully loaded on
first import and the import order is fixed in one place.
"""

from . import (
    _assessments,
    _chains,
    _coefficients,
    _covariances,
    _emitters,
    _engines,
    _filters,
    _fits,
    _inferences,
    _layouts,
    _levels,
    _means,
    _mixins,
    _models,
    _moments,
    _objectives,
    _parameters,
    _posteriors,
    _predictors,
    _priors,
    _results,
    _selections,
    _simulators,
    _smoothers,
    _solutions,
    _solvers,
    _specifications,
    _states,
    _substrates,
    _systems,
    _tests,
)

__all__ = [
    "_assessments",
    "_chains",
    "_coefficients",
    "_covariances",
    "_emitters",
    "_engines",
    "_filters",
    "_fits",
    "_inferences",
    "_layouts",
    "_levels",
    "_means",
    "_mixins",
    "_models",
    "_moments",
    "_objectives",
    "_parameters",
    "_posteriors",
    "_predictors",
    "_priors",
    "_results",
    "_selections",
    "_simulators",
    "_smoothers",
    "_solutions",
    "_solvers",
    "_specifications",
    "_states",
    "_substrates",
    "_systems",
    "_tests",
]
