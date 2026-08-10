from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _ParameterLayout:
    """Column bookkeeping for the responsibility-weighted regression.

    Switching and non-switching blocks share one design matrix; this object
    is the single source of truth for which column belongs to which regime,
    so the M-step never has to recompute offsets inline.
    """

    n_regimes: int
    order: int
    switching_mean: bool
    switching_ar: bool

    @property
    def n_intercept(self) -> int:
        """Number of intercept columns."""
        return self.n_regimes if self.switching_mean else 1

    @property
    def n_ar(self) -> int:
        """Number of autoregressive columns."""
        return self.n_regimes * self.order if self.switching_ar else self.order

    @property
    def width(self) -> int:
        """Total design width."""
        return self.n_intercept + self.n_ar

    def intercept_col(self, regime: int) -> int:
        """Index of the intercept column serving ``regime``."""
        return regime if self.switching_mean else 0

    def ar_slice(self, regime: int) -> slice:
        """Column slice of the AR block serving ``regime``."""
        base = self.n_intercept
        if self.switching_ar:
            return slice(base + regime * self.order, base + (regime + 1) * self.order)
        return slice(base, base + self.order)
