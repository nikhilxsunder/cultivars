import numpy as np
import numpy.typing as npt

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from .._core._validators import validate_choice, validate_order
from ..exceptions import SpecificationError
from ._models import _UnivariateModel

_VOL_FAMILIES = ("GARCH", "GJR", "EGARCH", "FIGARCH")
_MEANS = ("constant", "zero")

@dataclass(frozen=True, slots=True)
class _GARCHFit:
    """Raw outputs of a conditional-variance fit."""
    const: float | None
    ar_params: npt.NDArray[np.float64]
    omega: float
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    fractional_d: float | None
    conditional_variance: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int

class _GARCHModel[R](_UnivariateModel[R]):
    """Shared specification surface for the conditional-variance family.

    Holds the validation and parameter bookkeeping common to GARCH, GJR,
    EGARCH and FIGARCH. Public leaves bind ``R`` and assemble the result;
    they do not re-implement validation.
    """
    __slots__ = ("_vol", "_p", "_o", "_q", "_ar_lags", "_const", "_truncation")

    def __init__(
        self, endog: npt.ArrayLike, *, vol: str, p: int, o: int, q: int,
        ar_lags: int = 0, mean: str = "constant", truncation: int = 1000,
    ) -> None:
        super().__init__(endog)
        self._vol = validate_choice(vol, _VOL_FAMILIES, "vol")
        self._p = validate_order(p, "p")
        self._o = validate_order(o, "o")
        self._q = validate_order(q, "q")
        self._ar_lags = validate_order(ar_lags, "ar_lags")
        self._const = validate_choice(mean, _MEANS, "mean") == "constant"
        self._truncation = validate_order(truncation, "truncation", minimum=1)
        if self._vol == "GARCH" and self._o != 0:
            raise SpecificationError(
                "GARCH has no asymmetry term; set the asymmetry order o = 0."
            )
        if self._vol in ("GJR", "EGARCH") and self._o < 1:
            raise SpecificationError(f"{self._vol} requires an asymmetry order o >= 1.")
        self._ensure_length(
            max(self._p, self._o, self._q) + self._ar_lags + 2,
            f"{self._vol}({self._p}, {self._o}, {self._q})",
        )

    @property
    def vol(self) -> str:
        return self._vol
    @property
    def order(self) -> tuple[int, int, int]:
        return (self._p, self._o, self._q)

    def _fit_family(self) -> _GARCHFit:
        """Run the shared variance engine for this specification."""
        return _fit_garch_family(
            self.endog, vol=self._vol, p=self._p, o=self._o, q=self._q,
            ar_lags=self._ar_lags, include_const=self._const,
            truncation=self._truncation,
        )


def _backcast(resid: npt.NDArray[np.float64]) -> float:
    tau = min(75, resid.shape[0])
    w = 0.94 ** np.arange(tau)
    w /= w.sum()
    return float(np.sum(w * resid[:tau] ** 2))


def _garch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    sigma2 = np.empty(n)
    r2 = resid ** 2
    neg = (resid < 0.0).astype(np.float64)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * (r2[t - 1 - i] if t - 1 - i >= 0 else backcast)
        for k in range(o):
            if t - 1 - k >= 0:
                s += gamma[k] * r2[t - 1 - k] * neg[t - 1 - k]
            else:
                s += gamma[k] * backcast * 0.5
        for j in range(q):
            s += beta[j] * (sigma2[t - 1 - j] if t - 1 - j >= 0 else backcast)
        sigma2[t] = s
    return sigma2


def _egarch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    ln_sigma2 = np.empty(n)
    ln_bc = float(np.log(backcast))
    e = np.zeros(n)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * ((abs(e[t - 1 - i]) - _SQRT_2_PI) if t - 1 - i >= 0 else 0.0)
        for k in range(o):
            s += gamma[k] * (e[t - 1 - k] if t - 1 - k >= 0 else 0.0)
        for j in range(q):
            s += beta[j] * (ln_sigma2[t - 1 - j] if t - 1 - j >= 0 else ln_bc)
        ln_sigma2[t] = s
        sig = np.sqrt(np.exp(ln_sigma2[t]))
        e[t] = resid[t] / sig if sig > 0 else 0.0
    return np.exp(ln_sigma2)


def _figarch_weights(phi: float, d: float, beta: float, truncation: int) -> npt.NDArray[np.float64]:
    """ARCH(inf) lambda weights for FIGARCH(1, d, 1) (Chung 1999 recursion)."""
    lam = np.empty(truncation)
    delta = np.empty(truncation)
    lam[0] = phi - beta + d
    delta[0] = d
    for i in range(1, truncation):
        delta[i] = (i - d) / (i + 1) * delta[i - 1]
        lam[i] = beta * lam[i - 1] + (delta[i] - phi * delta[i - 1])
    return lam


def _figarch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    phi: float,
    d: float,
    beta: float,
    backcast: float,
    truncation: int = 1000,
) -> npt.NDArray[np.float64]:
    n = resid.shape[0]
    lam = _figarch_weights(phi, d, beta, truncation)
    r2 = resid ** 2
    sigma2 = np.empty(n)
    intercept = omega / (1.0 - beta)
    for t in range(n):
        m = min(t, truncation)
        acc = intercept
        if m > 0:
            acc += float(np.dot(lam[:m], r2[t - 1 :: -1][:m]))
        if truncation > m:
            acc += backcast * float(lam[m:truncation].sum())
        sigma2[t] = acc
    return sigma2


def _variance_block_size(vol: Vol, p: int, o: int, q: int) -> int:
    if vol == "FIGARCH":
        return 4
    if vol == "GARCH":
        return 1 + p + q
    return 1 + p + o + q            # GJR, EGARCH


def _unpack_variance(
    raw: npt.NDArray[np.float64], vol: Vol, p: int, o: int, q: int
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Return (omega, alpha, gamma, beta, frac_d); frac_d is ``nan`` unless FIGARCH."""
    if vol == "FIGARCH":
        omega = float(np.exp(raw[0]))
        fig_phi = _sig(float(raw[1]))
        fig_d = _sig(float(raw[2]))
        fig_beta = _sig(float(raw[3]))
        return omega, np.array([fig_phi]), np.zeros(0), np.array([fig_beta]), fig_d
    if vol == "GARCH":
        omega = float(np.exp(raw[0]))
        alpha = _softplus(raw[1 : 1 + p])
        gamma = np.zeros(0)
        beta = _softplus(raw[1 + p : 1 + p + q])
    elif vol == "GJR":
        omega = float(np.exp(raw[0]))
        alpha = _softplus(raw[1 : 1 + p])
        gamma = raw[1 + p : 1 + p + o]
        beta = _softplus(raw[1 + p + o : 1 + p + o + q])
    else:  # EGARCH
        omega = float(raw[0])
        alpha = raw[1 : 1 + p]
        gamma = raw[1 + p : 1 + p + o]
        beta = raw[1 + p + o : 1 + p + o + q]
    return omega, alpha, gamma, beta, float("nan")


def _variance_init(vol: Vol, var0: float, p: int, o: int, q: int) -> npt.NDArray[np.float64]:
    a_init, b_init, g_init = 0.05, 0.90, 0.05
    if vol == "FIGARCH":
        return np.array([np.log(var0 * 0.4), -1.0, -0.2, 0.4])
    if vol == "GARCH":
        return np.concatenate([
            [np.log(var0 * (1 - a_init - b_init))],
            [_inv_softplus(a_init)] * p,
            [_inv_softplus(b_init)] * q,
        ])
    if vol == "GJR":
        return np.concatenate([
            [np.log(var0 * (1 - a_init - b_init - 0.5 * g_init))],
            [_inv_softplus(a_init)] * p,
            [g_init] * o,
            [_inv_softplus(b_init)] * q,
        ])
    return np.concatenate([                      # EGARCH
        [np.log(var0) * (1 - 0.95)],
        [0.1] * p,
        [-0.05] * o,
        [0.95] * q,
    ])


def _conditional_variance(
    resid: npt.NDArray[np.float64],
    vol: Vol,
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    frac_d: float,
    backcast: float,
    truncation: int,
) -> npt.NDArray[np.float64]:
    if vol == "FIGARCH":
        return _figarch_variance(
            resid, omega, float(alpha[0]), frac_d, float(beta[0]), backcast, truncation
        )
    if vol == "EGARCH":
        return _egarch_variance(resid, omega, alpha, gamma, beta, backcast)
    return _garch_variance(resid, omega, alpha, gamma, beta, backcast)


def _persistence_ok(
    vol: Vol,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    frac_d: float,
    truncation: int,
) -> bool:
    if vol == "FIGARCH":
        from .garch import _figarch_weights

        lam = _figarch_weights(float(alpha[0]), frac_d, float(beta[0]), min(truncation, 200))
        return bool(np.all(lam >= -1e-6))
    if vol == "EGARCH":
        return abs(beta.sum()) < 0.999
    return alpha.sum() + 0.5 * gamma.sum() + beta.sum() < 0.999
