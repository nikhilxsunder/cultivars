# filepath: /src/cultivars/multivariate/structural/perturbation.py
#
# (MIT header)
"""Likelihood estimation of perturbation-solved DSGE models.

A DSGE model is a set of equilibrium conditions ``E_t f(y', y, x', x) =
0`` closed by a shock process; its solution is a pair of policy functions
``y = g(x)``, ``x' = h(x) + eta eps'``, and the solution *is* a state space
whose parameters are the structural parameters. Fernandez-Villaverde and
Rubio-Ramirez (2007) made the point that stopping at first order throws
away the model's nonlinearity and, with it, everything the likelihood
knows about risk; a second-order solution keeps the leading term of that
nonlinearity and the particle filter evaluates its likelihood without
further approximation.

That is the pipeline here. The specification supplies the equations,
the steady state, the shock loading, the measurement equation, and a
prior, all as functions of the structural parameters; the engine solves
the model to first or second order at any point (Schmitt-Grohe-Uribe
perturbation with numerical derivatives), emits the solution as a state
space on the substrate its order calls for -- exact linear-Gaussian at
first order, the pruned nonlinear system at second -- and estimates by
whichever likelihood that substrate offers:

- ``filter="kalman"``: exact, first order only. The Smets-Wouters
  tradition.
- ``filter="unscented"`` / ``"extended"``: a Gaussian approximation on
  the pruned second-order system, deterministic and fast; a good
  optimizer target and a poor final answer.
- ``filter="particle"``: simulated maximum likelihood under common
  random numbers, or -- through :meth:`PerturbationDSGE.sample` --
  particle marginal Metropolis-Hastings, the exact posterior at a
  particle filter per draw.

A first-order solution is a nested special case of the second-order
pruned system, so a specification estimated at ``order=1`` and at
``order=2`` can be compared on the same particle likelihood.

The equations are supplied as a plain callable and differentiated
numerically, so any model that can be written in numpy is admissible;
write it in logs for a log-linear approximation, in levels otherwise.

References:
    Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2007). Estimating
        macroeconomic models: A likelihood approach. *Review of Economic
        Studies*, 74(4), 1059-1087.
    Schmitt-Grohe, S., & Uribe, M. (2004). Solving dynamic general
        equilibrium models using a second-order approximation to the
        policy function. *Journal of Economic Dynamics and Control*,
        28(4), 755-775.
    Andreasen, M. M., Fernandez-Villaverde, J., & Rubio-Ramirez, J. F.
        (2018). The pruned state-space system for non-linear DSGE
        models: Theory and empirical applications. *Review of Economic
        Studies*, 85(1), 1-49.
    An, S., & Schorfheide, F. (2007). Bayesian analysis of DSGE models.
        *Econometric Reviews*, 26(2-4), 113-172.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable
from ..._internals import (
    _impulse_responses,
    _PerturbationDSGEPosterior,
    _PerturbationFit,
    _PerturbationModel,
    _PerturbationSolution,
    _SummaryMixin,
)
from ...exceptions import SpecificationError
from ...state_space import LinearGaussianSSM, NonlinearSSM

__all__ = [
    "PerturbationDSGE",
    "PerturbationDSGEResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class PerturbationDSGEResult(_SummaryMixin):
    """A point-estimated perturbation model.

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
    """

    data: npt.NDArray[np.float64] = field(repr=False)
    parameter_names: tuple[str, ...]
    theta: npt.NDArray[np.float64]
    solution: _PerturbationSolution = field(repr=False)
    llf: float
    filter: str
    order: int
    nobs: int
    n_params: int
    filtered_state: npt.NDArray[np.float64] = field(repr=False)
    _engine: PerturbationDSGE = field(repr=False)

    @classmethod
    def _from_fit(cls, fit: _PerturbationFit, model: PerturbationDSGE) -> PerturbationDSGEResult:
        """Assemble the public result from a raw fit and its specification."""
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
        """The estimate by name."""
        return {
            name: float(value) for name, value in zip(self.parameter_names, self.theta, strict=True)
        }

    @property
    def state_space(self) -> LinearGaussianSSM | NonlinearSSM:
        """The solution at the estimate on its substrate."""
        return self._engine.state_space(self.theta)

    def impulse_responses(
        self, *, horizon: int = 40, size: float = 1.0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Responses to each structural shock.

        Returns ``(states, controls)`` of shapes ``(n_eps, horizon, n_x)`` and
        ``(n_eps, horizon, n_y)``.

        Args:
            horizon: Periods after the impulse.
            size: Impulse size in standard deviations.

        Raises:
            SpecificationError: If the horizon is not positive.
        """
        if horizon < 1:
            raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
        return _impulse_responses(self.solution, horizon=horizon, size=size)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
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
    """A perturbation-solved DSGE model, estimated by likelihood.

    Args:
        specification: An object satisfying :class:`DSGESpecification`.
        data: The ``(nobs, p)`` observables, ``numpy.nan`` rows missing.
        order: Perturbation order, ``1`` or ``2``.

    Example:
        A stochastic growth model with log utility and full depreciation,
        whose exact policy is ``k' = alpha beta z k**alpha``::

            class Growth:
                parameter_names = ("alpha", "rho", "sigma")
                bounds = ((0.1, 0.6), (0.0, 0.999), (1e-4, 0.2))
                n_states, n_controls, beta = 2, 1, 0.99

                def equations(self, th, y_next, y, x_next, x):
                    a, rho, _ = th
                    (c,), (c_next,), (k, z), (k_next, z_next) = y, y_next, x, x_next
                    return np.array([
                        1 / c - self.beta / c_next * a * z_next * k_next ** (a - 1),
                        c + k_next - z * k ** a,
                        np.log(z_next) - rho * np.log(z),
                    ])

                def steady_state(self, th):
                    k = (th[0] * self.beta) ** (1 / (1 - th[0]))
                    return np.array([k, 1.0]), np.array([k ** th[0] - k])

                def shock_loading(self, th):
                    return np.array([[0.0], [th[2]]])

                def observation(self, th):
                    return np.array([[0.0, 0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])

                def log_prior(self, th):
                    return 0.0

            model = PerturbationDSGE(Growth(), consumption, order=2)
            res = model.fit(np.array([0.3, 0.9, 0.03]), filter="unscented")
            post = model.sample(res.theta, n_particles=500, n_draws=2000, n_burn=500)
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Structural parameter labels."""
        return tuple(self._spec.parameter_names)

    def solve(self, theta: npt.ArrayLike) -> _PerturbationSolution:
        """Solve the model at a parameter point.

        Args:
            theta: Structural parameters in the model's own space.

        Raises:
            NumericalError: If the steady state does not solve the
                equations or the Blanchard-Kahn conditions fail.
        """
        return self._solve(np.asarray(theta, dtype=np.float64))

    def state_space(self, theta: npt.ArrayLike) -> LinearGaussianSSM | NonlinearSSM:
        """The solution at ``theta`` on the substrate its order calls for."""
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

        Raises:
            SpecificationError: If ``theta0`` is missing or the filter is
                incompatible with the order.
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
