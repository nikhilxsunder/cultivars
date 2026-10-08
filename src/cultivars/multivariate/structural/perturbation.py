# filepath: /src/cultivars/multivariate/structural/perturbation.py
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
r"""Likelihood estimation of perturbation-solved DSGE models.

A DSGE model is a set of equilibrium conditions
:math:`\mathbb{E}_t f(y_{t+1}, y_t, x_{t+1}, x_t; \theta) = 0` closed by
a shock process, and its solution is a pair of policy functions

.. math::

   y_t = g(x_t; \theta), \qquad
   x_{t+1} = h(x_t; \theta) + \eta(\theta)\, \varepsilon_{t+1},

which *is* a state space whose parameters are the structural parameters.
Perturbation (Schmitt-Grohé and Uribe 2004) expands :math:`g` and
:math:`h` around the deterministic steady state: the first-order terms
solve a generalized eigenproblem under the Blanchard-Kahn condition, and
the second-order terms solve a linear system in the policy Hessians
whose risk correction :math:`g_{\sigma\sigma}, h_{\sigma\sigma}` is the
first place the model's attitude to uncertainty appears. Stopping at
first order throws that away and, with it, everything the likelihood
knows about risk; Fernández-Villaverde and Rubio-Ramírez (2007) made the
case that the second-order solution keeps the leading term and the
particle filter evaluates its likelihood without further approximation.
The second-order system is pruned (Andreasen, Fernández-Villaverde and
Rubio-Ramírez 2018) so that it has finite moments and cannot explode
along a sample path, which is what makes its likelihood well defined.

That is the pipeline here. The specification supplies the equations, the
steady state, the shock loading, the measurement equation and a log
prior, all as functions of :math:`\theta`; the engine solves the model
at any point by numerical differentiation of the equations, emits the
solution as a state space on the substrate its order calls for -- exact
linear-Gaussian at first order, the pruned nonlinear system at second --
and estimates by whichever likelihood that substrate offers. Under
``filter="kalman"`` the criterion is exact and available at first order
only, the Smets-Wouters tradition. Under ``"extended"`` or
``"unscented"`` it is a Gaussian approximation on the pruned
second-order system, deterministic and cheap: a good optimizer target
and a poor final answer. Under ``"particle"`` it is a bootstrap-filter
estimate under common random numbers, maximized by Nelder-Mead, or --
through :meth:`PerturbationDSGE.sample` -- the exact posterior by
particle marginal Metropolis-Hastings with one filter per draw. A
first-order solution is the second-order pruned system with its
quadratic blocks zero, so the same specification at ``order=1`` and
``order=2`` can be compared on the same particle likelihood.

Two commitments shape the surface. First, the model is the user's and
only the user's. The equations are a plain callable differentiated
numerically, so any model that can be written in numpy is admissible --
in logs for a log-linear approximation, in levels otherwise -- and the
engine never reorders, renames or rescales what it is given; parameters
travel through the optimizer in an unconstrained transform of the
declared bounds and come back in the model's own space. Second, the
criterion is named rather than blurred. A result records which filter
produced its value, the summary says in one sentence what that value is,
and values across filters are not presented as comparable, because they
are not: a particle estimate with too few particles is biased downward,
severely so when measurement noise is small relative to the state
innovations, and the deterministic filters are approximations whose
error the module does not pretend to bound.

Layout. :class:`PerturbationDSGE` is a ``_PerturbationModel`` from
``_internals`` against the ``_PerturbationModelSpecification`` protocol
there; the solver is ``_solve_perturbation`` in ``_internals`` over
``_first_order`` and ``_second_order`` in ``_core``, the substrate is
emitted by ``_pruned_state_space``, the point estimate comes from
``_maximize_likelihood`` on a ``_PerturbationObjective`` or a
``_ParticleLikelihoodObjective``, and the posterior from
``_ParticleMarginalChain``. :class:`PerturbationDSGEResult` carries the
estimate and the solution and re-emits the latter through ``_impulse_responses``
and ``_simulate_perturbation``. The filters themselves live in
:mod:`~cultivars.state_space.linear_gaussian` and
:mod:`~cultivars.state_space.nonlinear`.

References:
    Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007). Estimating
    macroeconomic models: A likelihood approach. *Review of Economic
    Studies*, 74(4), 1059-1087.

    Schmitt-Grohé, S., & Uribe, M. (2004). Solving dynamic general
    equilibrium models using a second-order approximation to the policy
    function. *Journal of Economic Dynamics and Control*, 28(4), 755-775.

    Andreasen, M. M., Fernández-Villaverde, J., & Rubio-Ramírez, J. F.
    (2018). The pruned state-space system for non-linear DSGE models:
    Theory and empirical applications. *Review of Economic Studies*,
    85(1), 1-49.

    An, S., & Schorfheide, F. (2007). Bayesian analysis of DSGE models.
    *Econometric Reviews*, 26(2-4), 113-172.

Example:
    The same latent AR(1) model solved at both orders. The first-order
    exact likelihood and the second-order Gaussian approximation agree
    on a linear model, the first-order fit is the second-order fit, and
    the second-order solution's quadratic and risk blocks are zero:

    >>> import numpy as np
    >>> class Latent:
    ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
    ...     n_states, n_controls = 1, 1
    ...     def equations(self, th, y_next, y, x_next, x):
    ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
    ...     def steady_state(self, th):
    ...         return np.zeros(1), np.zeros(1)
    ...     def shock_loading(self, th):
    ...         return np.array([[th[1]]])
    ...     def observation(self, th):
    ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
    ...     def log_prior(self, th):
    ...         return 0.0
    >>> rng = np.random.default_rng(0)
    >>> x = np.zeros(200)
    >>> for t in range(1, 200):
    ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
    >>> first = PerturbationDSGE(Latent(), x, order=1).fit([0.5, 0.3], filter="kalman")
    >>> second = PerturbationDSGE(Latent(), x, order=2).fit([0.5, 0.3])
    >>> first.filter, second.filter, bool(abs(first.llf - second.llf) < 1.0)
    ('kalman', 'unscented', True)
    >>> bool(np.allclose(first.theta, second.theta, atol=1e-2))
    True
    >>> quadratic = second.solution
    >>> bool(np.allclose(quadratic.h_xx, 0.0)), bool(np.allclose(quadratic.h_ss, 0.0))
    (True, True)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import _SIMULATION_BURN, SummaryTable
from ..._internals import (
    _impulse_responses,
    _PerturbationDSGEPosterior,
    _PerturbationFit,
    _PerturbationModel,
    _PerturbationSolution,
    _simulate_perturbation,
    _SummaryMixin,
)
from ...exceptions import SpecificationError
from ...state_space.linear_gaussian import LinearGaussianSSM
from ...state_space.nonlinear import NonlinearSSM

__all__ = [
    "PerturbationDSGE",
    "PerturbationDSGEResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class PerturbationDSGEResult(_SummaryMixin):
    r"""A point-estimated perturbation model.

    The structural parameters :math:`\theta` that maximize one filter's
    likelihood of the observables under the perturbation solution

    .. math::

       x_{t+1} = h(x_t; \theta) + \eta(\theta)\, \varepsilon_{t+1},
       \qquad
       y_t = g(x_t; \theta),
       \qquad
       z_t = Z\,(x_t, y_t) + d + e_t,

    with :math:`h` and :math:`g` the first- or pruned second-order policy
    functions and :math:`z_t` the observables. The record carries the
    estimate, the solution at it, the criterion value, the filter that
    produced it and the filtered state path; the engine is kept so the
    solution can be re-emitted on its substrate and fresh samples drawn.

    Note:
        :attr:`llf` is whichever criterion was maximized and only that:
        exact under ``"kalman"``, the extended or unscented filter's
        Gaussian approximation on the pruned second-order system, or a
        particle estimate under common random numbers. Values across
        filters are not comparable, and a particle value with few
        particles is biased downward -- severely so when the measurement
        noise is small relative to the state innovations, because the
        bootstrap filter degenerates. No standard errors are carried: the
        inferential surface of this family is :meth:`PerturbationDSGE.sample`.

    Attributes:
        data: The observables.
        parameter_names: Structural parameter labels.
        theta: The estimate, in the model's own space.
        solution: The perturbation solution at ``theta``.
        llf: The criterion at the optimum: exact under ``"kalman"``, a
            Gaussian approximation under ``"extended"`` / ``"unscented"``,
            a particle estimate under ``"particle"``.
        filter: Which filter produced ``llf``.
        order: Perturbation order.
        nobs: Observations.
        n_params: Structural parameters.
        filtered_state: Filtered state means.

    See Also:
        * :class:`PerturbationDSGE` -- the model whose ``fit()`` returns
          this record and whose ``sample()`` gives the posterior.
        * :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`,
          :class:`~cultivars.state_space.nonlinear.NonlinearSSM` -- the
          substrates :attr:`state_space` lands on.

    References:
        Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007). Estimating
        macroeconomic models: A likelihood approach. *Review of Economic
        Studies*, 74(4), 1059-1087.

        Schmitt-Grohé, S., & Uribe, M. (2004). Solving dynamic general
        equilibrium models using a second-order approximation to the policy
        function. *Journal of Economic Dynamics and Control*, 28(4),
        755-775.

    Example:
        The smallest admissible specification: one state following an
        AR(1), one control equal to it, observed with negligible noise. The
        exact first-order fit recovers the persistence and the innovation
        scale, and the solution at the estimate is the AR(1) itself:

        >>> import numpy as np
        >>> class Latent:
        ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
        ...     n_states, n_controls = 1, 1
        ...     def equations(self, th, y_next, y, x_next, x):
        ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
        ...     def steady_state(self, th):
        ...         return np.zeros(1), np.zeros(1)
        ...     def shock_loading(self, th):
        ...         return np.array([[th[1]]])
        ...     def observation(self, th):
        ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
        ...     def log_prior(self, th):
        ...         return 0.0
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(200)
        >>> for t in range(1, 200):
        ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
        >>> res = PerturbationDSGE(Latent(), x, order=1).fit([0.5, 0.3], filter="kalman")
        >>> res.filter, res.order, res.nobs, res.n_params, res.parameter_names
        ('kalman', 1, 200, 2, ('rho', 'sigma'))
        >>> {name: round(value, 2) for name, value in res.params.items()}
        {'rho': 0.82, 'sigma': 0.48}
        >>> res.solution.h_x.round(2).tolist(), res.solution.is_stable, res.filtered_state.shape
        ([[0.82]], True, (200, 1))
    """

    data: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, p)`` observables as supplied, ``nan`` rows missing. Kept out of the repr."""

    parameter_names: tuple[str, ...]
    """Structural parameter labels, in the specification's order."""

    theta: npt.NDArray[np.float64]
    """The ``(n_params,)`` estimate, in the model's own (constrained) space."""

    solution: _PerturbationSolution = field(repr=False)
    """The perturbation solution at :attr:`theta`. Kept out of the repr."""

    llf: float
    """The maximized criterion; exact only under ``"kalman"``."""

    filter: str
    """``"kalman"``, ``"extended"``, ``"unscented"`` or ``"particle"``."""

    order: int
    """Perturbation order, ``1`` or ``2``."""

    nobs: int
    """Observations, including rows that are ``nan``."""

    n_params: int
    """Structural parameters."""

    filtered_state: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, n_x)`` filtered state means. Kept out of the repr."""

    _engine: PerturbationDSGE = field(repr=False)
    """The model that produced the fit, kept for re-emission and simulation.

    Kept out of the repr.
    """

    @classmethod
    def _from_fit(cls, fit: _PerturbationFit, model: PerturbationDSGE) -> PerturbationDSGEResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed fit: estimate, solution, criterion, filter name,
                effective sample, parameter count and filtered states.
            model: The model that produced it, kept on the result as its
                engine.

        Returns:
            A populated result.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> model = PerturbationDSGE(Latent(), rng.standard_normal(80), order=1)
            >>> fit = model._fit_point(
            ...     np.array([0.5, 0.5]), filter_name="kalman", n_particles=0, seed=None
            ... )
            >>> res = PerturbationDSGEResult._from_fit(fit, model)
            >>> res._engine is model, res.filter, res.order
            (True, 'kalman', 1)
        """
        return cls(
            data=model.data,
            parameter_names=tuple(model.parameter_names),
            theta=fit.theta,
            solution=fit.solution,
            llf=fit.llf,
            filter=fit.filter,
            order=model.order,
            nobs=fit.nobs,
            n_params=fit.n_params,
            filtered_state=fit.filtered_state,
            _engine=model,
        )

    @property
    def params(self) -> dict[str, float]:
        """The estimate by name.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> res = PerturbationDSGE(Latent(), rng.standard_normal(80), order=1).fit([0.5, 0.5])
            >>> list(res.params), bool(np.allclose(list(res.params.values()), res.theta))
            (['rho', 'sigma'], True)
        """
        return {
            name: float(value) for name, value in zip(self.parameter_names, self.theta, strict=True)
        }

    @property
    def state_space(self) -> LinearGaussianSSM | NonlinearSSM:
        """The solution at the estimate on its substrate.

        Linear-Gaussian at first order, the pruned nonlinear system at
        second; re-emitted through the engine so it is the same object the
        likelihood saw.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal(80)
            >>> first = PerturbationDSGE(Latent(), x, order=1).fit([0.5, 0.5], filter="kalman")
            >>> second = PerturbationDSGE(Latent(), x, order=2).fit([0.5, 0.5])
            >>> type(first.state_space).__name__, type(second.state_space).__name__
            ('_LinearGaussianStateSpace', '_NonlinearStateSpace')
        """
        return self._engine.state_space(self.theta)

    def impulse_responses(
        self, *, horizon: int = 40, size: float = 1.0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Responses to each structural shock.

        Deterministic paths from the steady state after a one-time impulse
        of ``size`` standard deviations in one structural shock, through
        the solution at the estimate.

        Args:
            horizon: Periods after the impulse.
            size: Impulse size in standard deviations.

        Returns:
            ``(states, controls)`` of shapes ``(n_eps, horizon, n_x)`` and
            ``(n_eps, horizon, n_y)``, as deviations from the steady state.

        Raises:
            SpecificationError: If the horizon is not positive.

        Example:
            For the AR(1) latent state the response decays geometrically at
            the estimated persistence:

            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> x = np.zeros(200)
            >>> for t in range(1, 200):
            ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
            >>> res = PerturbationDSGE(Latent(), x, order=1).fit([0.5, 0.3], filter="kalman")
            >>> states, controls = res.impulse_responses(horizon=3)
            >>> states.shape, controls.shape
            ((1, 3, 1), (1, 3, 1))
            >>> rho, sigma = res.theta
            >>> bool(np.allclose(states[0, :, 0], sigma * rho ** np.arange(3)))
            True
            >>> res.impulse_responses(horizon=0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be at least 1; got 0.
        """
        if horizon < 1:
            raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
        return _impulse_responses(self.solution, horizon=horizon, size=size)

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample of the observables at the estimate.

        Standard-normal structural shocks drive the pruned solution from
        the steady state, the first ``burn`` periods are discarded, and the
        observables are read through the model's measurement equation
        with its stated measurement noise.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n, p)`` in the order of the data columns.

        Raises:
            SpecificationError: If the counts are not usable.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> res = PerturbationDSGE(Latent(), rng.standard_normal(80), order=1).fit([0.5, 0.5])
            >>> sample = res.simulate(50, seed=1)
            >>> sample.shape, bool(np.allclose(sample, res.simulate(50, seed=1)))
            ((50, 1), True)
            >>> res.simulate(0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n must be positive and burn non-negative; ...
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        design, intercept, obs_cov = self._engine._measurement(self.theta, self.solution)
        return _simulate_perturbation(
            self.solution,
            n,
            design=design,
            intercept=intercept,
            obs_cov=obs_cov,
            rng=rng,
            burn=burn,
        )

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per structural parameter; the filter, criterion value and
        system dimensions in the metadata, with the first-order stability
        verdict; one note naming exactly what the criterion is.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> res = PerturbationDSGE(Latent(), rng.standard_normal(80), order=2).fit([0.5, 0.5])
            >>> table = res._summary_table()
            >>> table.title, [row[0] for row in table.rows], dict(table.metadata)["First order"]
            ('Perturbation DSGE (order 2) Results', ['rho', 'sigma'], 'stable')
            >>> table.notes[0][:40]
            'Unscented-filter Gaussian approximation '
        """
        rows = tuple((name, f"{value:.6g}") for name, value in self.params.items())
        criterion = {
            "kalman": "Exact Gaussian likelihood of the first-order solution.",
            "extended": "Extended-filter Gaussian approximation on the pruned second-order system.",
            "unscented": (
                "Unscented-filter Gaussian approximation on the pruned second-order system."
            ),
            "particle": "Particle estimate of the exact likelihood under common random numbers.",
        }[self.filter]
        stability = "stable" if self.solution.is_stable else "UNSTABLE"
        return SummaryTable(
            title=f"Perturbation DSGE (order {self.order}) Results",
            metadata=(
                ("Filter", self.filter),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("States / controls", f"{self.solution.n_states} / {self.solution.n_controls}"),
                ("Shocks", f"{self.solution.n_shocks}"),
                ("First order", stability),
            ),
            columns=("", "estimate"),
            rows=rows,
            notes=(criterion,),
        )


class PerturbationDSGE(_PerturbationModel[PerturbationDSGEResult]):
    r"""A perturbation-solved DSGE model, estimated by likelihood.

    The specification supplies the equilibrium conditions
    :math:`\mathbb{E}_t f(y_{t+1}, y_t, x_{t+1}, x_t; \theta) = 0`, the
    steady state, the shock loading, the measurement equation and a log
    prior, all as functions of the structural parameters; the engine solves
    the model at any :math:`\theta` to first or second order by
    Schmitt-Grohé-Uribe perturbation with numerical derivatives, emits the
    solution as a state space on the substrate the order calls for, and
    estimates by whichever likelihood that substrate offers. The
    constructor is inherited from ``_PerturbationModel`` and validates the
    order, the data and the specification's bounds; the parameters travel
    through the optimizer in an unconstrained transform of those bounds.

    Args:
        specification: An object with ``parameter_names``, ``bounds``,
            ``n_states``, ``n_controls`` and the methods ``equations``,
            ``steady_state``, ``shock_loading``, ``observation`` and
            ``log_prior``; the protocol is ``_PerturbationModelSpecification``
            in ``_internals``.
        data: The ``(nobs, p)`` observables, ``numpy.nan`` rows missing.
        order: Perturbation order, ``1`` or ``2``.

    Raises:
        SpecificationError: If ``order`` is not 1 or 2, the bounds are
            malformed, or the parameter names and bounds disagree in length.
        DimensionError: If ``data`` is not a matrix.
        NumericalError: If ``data`` contains infinities.

    Attributes:
        _spec: The specification.
        _data: The validated ``(nobs, p)`` observables.
        _order: The perturbation order.
        _bounds: The ``(n_params, 2)`` parameter bounds.

    See Also:
        * :class:`PerturbationDSGEResult` -- the record ``fit()`` returns.
        * :mod:`~cultivars.state_space.nonlinear` -- the filters the
          second-order likelihood is evaluated through.

    References:
        Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007). Estimating
        macroeconomic models: A likelihood approach. *Review of Economic
        Studies*, 74(4), 1059-1087.

        An, S., & Schorfheide, F. (2007). Bayesian analysis of DSGE models.
        *Econometric Reviews*, 26(2-4), 113-172.

    Example:
        A stochastic growth model with log utility and full depreciation,
        whose exact policy is ``k' = alpha beta z k**alpha``. Consumption is
        simulated from that policy, the second-order model is fitted through
        the unscented filter, and the solution's first-order block carries
        the capital-accumulation and productivity dynamics:

        >>> import numpy as np
        >>> class Growth:
        ...     parameter_names = ("alpha", "rho", "sigma")
        ...     bounds = ((0.1, 0.6), (0.0, 0.999), (1e-4, 0.2))
        ...     n_states, n_controls, beta = 2, 1, 0.99
        ...     def equations(self, th, y_next, y, x_next, x):
        ...         a, rho, _ = th
        ...         (c,), (c_next,), (k, z), (k_next, z_next) = y, y_next, x, x_next
        ...         return np.array([
        ...             1 / c - self.beta / c_next * a * z_next * k_next ** (a - 1),
        ...             c + k_next - z * k ** a,
        ...             np.log(z_next) - rho * np.log(z),
        ...         ])
        ...     def steady_state(self, th):
        ...         k = (th[0] * self.beta) ** (1 / (1 - th[0]))
        ...         return np.array([k, 1.0]), np.array([k ** th[0] - k])
        ...     def shock_loading(self, th):
        ...         return np.array([[0.0], [th[2]]])
        ...     def observation(self, th):
        ...         return np.array([[0.0, 0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
        ...     def log_prior(self, th):
        ...         return 0.0
        >>> rng = np.random.default_rng(0)
        >>> alpha, rho, sigma, beta = 0.33, 0.9, 0.03, 0.99
        >>> k, z, consumption = (alpha * beta) ** (1 / (1 - alpha)), 1.0, []
        >>> for t in range(300):
        ...     z = np.exp(rho * np.log(z) + sigma * rng.standard_normal())
        ...     output = z * k ** alpha
        ...     k = alpha * beta * output
        ...     consumption.append(output - k)
        >>> model = PerturbationDSGE(Growth(), np.array(consumption[100:]), order=2)
        >>> res = model.fit([0.3, 0.8, 0.05], filter="unscented")
        >>> {name: round(value, 2) for name, value in res.params.items()}
        {'alpha': 0.35, 'rho': 0.9, 'sigma': 0.02}
        >>> own = np.diag(res.solution.h_x)
        >>> bool(np.allclose(own, [res.params["alpha"], res.params["rho"]])), res.solution.is_stable
        (True, True)
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Structural parameter labels.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> PerturbationDSGE(Latent(), np.zeros(50), order=1).parameter_names
            ('rho', 'sigma')
        """
        return tuple(self._spec.parameter_names)

    def solve(self, theta: npt.ArrayLike) -> _PerturbationSolution:
        """Solve the model at a parameter point.

        Args:
            theta: Structural parameters in the model's own space.

        Returns:
            The ``_PerturbationSolution`` with the policy blocks ``h_x``,
            ``g_x`` (and ``h_xx``, ``g_xx``, ``h_ss``, ``g_ss`` at second
            order), the loading ``eta`` and the steady state.

        Raises:
            NumericalError: If the steady state does not solve the
                equations or the Blanchard-Kahn conditions fail.

        Example:
            For the AR(1) latent state the first-order solution is the
            model itself:

            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> solution = PerturbationDSGE(Latent(), np.zeros(50), order=1).solve([0.8, 0.5])
            >>> solution.h_x.tolist(), solution.g_x.tolist(), solution.eta.tolist()
            ([[0.8]], [[1.0]], [[0.5]])
            >>> solution.order, solution.is_stable, solution.n_shocks
            (1, True, 1)
        """
        return self._solve(np.asarray(theta, dtype=np.float64))

    def state_space(self, theta: npt.ArrayLike) -> LinearGaussianSSM | NonlinearSSM:
        """The solution at ``theta`` on the substrate its order calls for.

        Args:
            theta: Structural parameters in the model's own space.

        Returns:
            A linear-Gaussian system at first order; the pruned nonlinear
            system at second.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> first = PerturbationDSGE(Latent(), np.zeros(50), order=1).state_space([0.8, 0.5])
            >>> second = PerturbationDSGE(Latent(), np.zeros(50), order=2).state_space([0.8, 0.5])
            >>> type(first).__name__, type(second).__name__
            ('_LinearGaussianStateSpace', '_NonlinearStateSpace')
        """
        return self._state_space(np.asarray(theta, dtype=np.float64))

    def loglikelihood(
        self,
        theta: npt.ArrayLike,
        *,
        filter: str = "unscented",
        n_particles: int = 1000,
        seed: int | None = None,
    ) -> float:
        """The log-likelihood (or its particle estimate) at one parameter point.

        Args:
            theta: Structural parameters.
            filter: ``"kalman"`` (first order only), ``"extended"``,
                ``"unscented"``, or ``"particle"``.
            n_particles: Particles under ``"particle"``.
            seed: Seed under ``"particle"``.

        Returns:
            The criterion value; exact only under ``"kalman"``.

        Example:
            On a linear model the first-order exact likelihood, the
            second-order unscented approximation and a well-resourced
            particle estimate agree; a starved particle filter does not:

            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> x = np.zeros(200)
            >>> for t in range(1, 200):
            ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
            >>> first = PerturbationDSGE(Latent(), x, order=1)
            >>> exact = first.loglikelihood([0.8, 0.5], filter="kalman")
            >>> second = PerturbationDSGE(Latent(), x, order=2)
            >>> approx = second.loglikelihood([0.8, 0.5])
            >>> rich = second.loglikelihood([0.8, 0.5], filter="particle", n_particles=5000, seed=0)
            >>> poor = second.loglikelihood([0.8, 0.5], filter="particle", n_particles=200, seed=0)
            >>> bool(abs(approx - exact) < 1.0), bool(abs(rich - exact) < 1.0)
            (True, True)
            >>> bool(poor < exact - 50)
            True
        """
        return self._loglikelihood(
            np.asarray(theta, dtype=np.float64),
            filter_name=filter,
            n_particles=n_particles,
            seed=seed,
        )

    def fit(
        self,
        theta0: npt.ArrayLike | None = None,
        *,
        filter: str = "unscented",
        n_particles: int = 500,
        seed: int | None = None,
    ) -> PerturbationDSGEResult:
        """Maximize one filter's likelihood from a starting point.

        Args:
            theta0: Starting structural parameters. Required: a DSGE
                likelihood has no data-driven warm start.
            filter: ``"kalman"`` (exact, first order only), ``"extended"``,
                ``"unscented"``, or ``"particle"`` (common random numbers,
                Nelder-Mead).
            n_particles: Particles under ``"particle"``.
            seed: The common seed under ``"particle"``.

        Returns:
            The :class:`PerturbationDSGEResult` at the optimum.

        Raises:
            SpecificationError: If ``theta0`` is missing or the filter is
                incompatible with the order.

        Example:
            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> x = np.zeros(200)
            >>> for t in range(1, 200):
            ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
            >>> second = PerturbationDSGE(Latent(), x, order=2)
            >>> res = second.fit([0.5, 0.3], filter="extended")
            >>> res.filter, {name: round(value, 2) for name, value in res.params.items()}
            ('extended', {'rho': 0.82, 'sigma': 0.48})
            >>> second.fit()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: theta0 is required: supply starting ...
            >>> second.fit([0.5, 0.3], filter="kalman")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: filter='kalman' is exact only at order=1; ...
        """
        if theta0 is None:
            raise SpecificationError(
                "theta0 is required: supply starting structural parameters inside the bounds."
            )
        if filter == "kalman" and self.order != 1:
            raise SpecificationError(
                "filter='kalman' is exact only at order=1; construct the model with "
                "order=1 or choose 'extended', 'unscented', or 'particle'."
            )
        fit = self._fit_point(
            np.asarray(theta0, dtype=np.float64),
            filter_name=filter,
            n_particles=n_particles,
            seed=seed,
        )
        return PerturbationDSGEResult._from_fit(fit, self)

    def sample(
        self,
        theta0: npt.ArrayLike,
        *,
        n_particles: int = 500,
        n_draws: int = 2000,
        n_burn: int = 500,
        thin: int = 1,
        proposal_scale: npt.ArrayLike | None = None,
        seed: int | None = None,
    ) -> _PerturbationDSGEPosterior:
        """Particle marginal Metropolis-Hastings under the specification's prior.

        Each iteration proposes in the unconstrained space, estimates the
        likelihood with a fresh particle filter, and accepts by the
        pseudo-marginal ratio, which targets the exact posterior whatever
        the particle count; the count only governs mixing.

        Args:
            theta0: Starting structural parameters; a point estimate from
                :meth:`fit` is the natural choice.
            n_particles: Particles per likelihood estimate. Pitt et al.
                (2012) suggest sizing it so the log-likelihood estimate's
                standard deviation at the mode is near one.
            n_draws: Total iterations.
            n_burn: Burn-in, during which the proposal adapts.
            thin: Keep every ``thin``-th post-burn draw.
            proposal_scale: Initial proposal standard deviations in the
                *unconstrained* space, ``(d,)``, or a covariance
                ``(d, d)``; ``None`` for ``0.1`` per coordinate.
            seed: Seed.

        Returns:
            The ``_PerturbationDSGEPosterior`` with the kept draws, their
            log-likelihood and log-posterior values, the acceptance rate
            and convergence diagnostics.

        Example:
            A short chain, far too short for inference, shows the shape of
            the surface:

            >>> import numpy as np
            >>> class Latent:
            ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
            ...     n_states, n_controls = 1, 1
            ...     def equations(self, th, y_next, y, x_next, x):
            ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
            ...     def steady_state(self, th):
            ...         return np.zeros(1), np.zeros(1)
            ...     def shock_loading(self, th):
            ...         return np.array([[th[1]]])
            ...     def observation(self, th):
            ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
            ...     def log_prior(self, th):
            ...         return 0.0
            >>> rng = np.random.default_rng(0)
            >>> x = np.zeros(120)
            >>> for t in range(1, 120):
            ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
            >>> model = PerturbationDSGE(Latent(), x, order=2)
            >>> res = model.fit([0.5, 0.3])
            >>> post = model.sample(res.theta, n_particles=100, n_draws=40, n_burn=10, seed=0)
            >>> post.n_kept, post.n_particles, bool(0.0 <= post.acceptance_rate <= 1.0)
            (30, 100, True)
        """
        fit = self._sample_chain(
            np.asarray(theta0, dtype=np.float64),
            n_particles=n_particles,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            proposal_scale=proposal_scale,
            seed=seed,
        )
        return _PerturbationDSGEPosterior._from_fit(fit, self)
