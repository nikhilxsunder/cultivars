# filepath: /src/cultivars/engine/_internals/_specifications.py
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

"""Model specifications: what a user-written model must supply to the engine.

One protocol, :class:`_PerturbationModelSpecification`, the contract a
rational-expectations model satisfies to be solved by perturbation and
estimated. The user writes the equilibrium conditions, the steady state,
the shock loading, the measurement equation, and the prior as plain methods
on a class; the engine differentiates, solves, filters, and samples. The
protocol is structural, so the class inherits from nothing.

The solver that consumes a specification is ``_solve_perturbation`` in
``_solvers``, the estimation engine is ``_PerturbationModel`` in
``_models``, and the public front is
:class:`cultivars.multivariate.structural.perturbation.PerturbationDSGE`.

Background: perturbation solution of rational-expectations models, to
second order.

A model is a system of equilibrium conditions ``E_t f(y', y, x', x) = 0``
with ``x`` the predetermined states (including any exogenous driving
processes) and ``y`` the controls, closed by ``x' = h(x, sigma) + sigma
eta eps'``. Schmitt-Grohe and Uribe (2004) showed that the Taylor
coefficients of the policy functions ``g`` and ``h`` around the
deterministic steady state solve a sequence of *linear* problems once the
first-order coefficients are known: the first order is the Blanchard-Kahn
/ Klein generalized-Schur problem, the second order in ``x`` is one linear
system in ``(g_xx, h_xx)``, and the risk correction is one linear system in
``(g_ss, h_ss)`` -- with the cross terms ``g_xs``, ``h_xs`` identically
zero.

Derivatives of ``f`` are taken numerically (central differences), so the
model is supplied as a plain callable and no symbolic layer is needed;
the cost is a step-size floor on accuracy of order ``1e-8`` in the second
derivatives, immaterial next to the perturbation error itself.

The second-order solution is carried as the *pruned* state space of Kim,
Kim, Schaumburg, and Sims (2008) in the form of Andreasen,
Fernandez-Villaverde, and Rubio-Ramirez (2018): a first-order state and a
second-order state evolve side by side, the quadratic term is fed only by
the first-order state, and the system cannot explode. That is what makes
it a well-posed customer of the nonlinear filters.

References:
    Schmitt-Grohe, S., & Uribe, M. (2004). Solving dynamic general
        equilibrium models using a second-order approximation to the
        policy function. *Journal of Economic Dynamics and Control*,
        28(4), 755-775.
    Klein, P. (2000). Using the generalized Schur form to solve a
        multivariate linear rational expectations model. *Journal of
        Economic Dynamics and Control*, 24(10), 1405-1423.
    Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). Calculating
        and using second-order accurate solutions of discrete time
        dynamic equilibrium models. *Journal of Economic Dynamics and
        Control*, 32(11), 3397-3414.
    Andreasen, M. M., Fernandez-Villaverde, J., & Rubio-Ramirez, J. F.
        (2018). The pruned state-space system for non-linear DSGE
        models: Theory and empirical applications. *Review of Economic
        Studies*, 85(1), 1-49.
    Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2007). Estimating
        macroeconomic models: A likelihood approach. *Review of Economic
        Studies*, 74(4), 1059-1087.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "_PerturbationModelSpecification",
]


class _PerturbationModelSpecification(Protocol):
    """What a perturbation model must supply to be solved and estimated.

    Every method takes the structural parameter vector ``theta`` in its
    natural (constrained) space. The engine differentiates ``equations``
    numerically, so it must be smooth in its arguments near the steady
    state and must be written in the variables the approximation should
    be taken in -- write it in logs for a log-linear approximation.

    Attributes:
        parameter_names: One label per structural parameter.
        bounds: ``(d, 2)`` lower and upper bounds per parameter;
            ``-inf`` / ``inf`` for an unbounded side.
        n_states: Predetermined variables, exogenous processes included.
        n_controls: Non-predetermined variables.

    Note:
        The protocol is not ``runtime_checkable``, and nothing here checks
        an implementation against it. A method with the wrong signature or
        a return of the wrong shape surfaces where the engine first uses
        it.
    """

    parameter_names: Sequence[str]
    """One label per structural parameter, in the order of ``theta``."""
    bounds: Sequence[tuple[float, float]]
    """``(lower, upper)`` per parameter; ``-inf`` or ``inf`` for an unbounded side.

    The engine maps each bounded parameter to an unconstrained coordinate,
    so the bounds define the space the optimizer and the chain move in.
    """
    n_states: int
    """Number of predetermined variables ``n_x``, exogenous processes included."""
    n_controls: int
    """Number of non-predetermined variables ``n_y``."""

    def equations(
        self,
        theta: npt.NDArray[np.float64],
        y_next: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_next: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Residuals of ``E_t f(y', y, x', x) = 0``, ``n_states + n_controls`` of them.

        Called many times with perturbed arguments to build numerical
        derivatives, so it must be smooth near the steady state and free
        of side effects.

        Args:
            theta: Structural parameters in the constrained space.
            y_next: Controls next period, ``(n_controls,)``.
            y: Controls this period, ``(n_controls,)``.
            x_next: States next period, ``(n_states,)``.
            x: States this period, ``(n_states,)``.

        Returns:
            The residual vector, ``(n_states + n_controls,)``; zero at the
            steady state.
        """
        ...

    def steady_state(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(x_ss, y_ss)`` solving the equations with expectations removed.

        Args:
            theta: Structural parameters in the constrained space.

        Returns:
            The steady-state states, ``(n_states,)``, and controls,
            ``(n_controls,)``.
        """
        ...

    def shock_loading(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``(n_states, n_shocks)`` loading of standard-normal shocks on the states.

        Args:
            theta: Structural parameters in the constrained space.

        Returns:
            The loading matrix ``eta``. A row of zeros marks a state no
            shock hits directly.
        """
        ...

    def observation(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(design, intercept, obs_cov)``: observables ``= design @ [x; y] + intercept + e``.

        Args:
            theta: Structural parameters in the constrained space.

        Returns:
            The ``(p, n_states + n_controls)`` design over the stacked
            states and controls, the ``(p,)`` intercept, and the ``(p, p)``
            covariance of the measurement error ``e``.
        """
        ...

    def log_prior(self, theta: npt.NDArray[np.float64]) -> float:
        """Log prior density in the constrained space; ``-inf`` outside support.

        Args:
            theta: Structural parameters in the constrained space.

        Returns:
            The log density, which need not be normalized. The engine adds
            the log-Jacobian of its own reparameterization.
        """
        ...
