import numpy as np
import numpy.typing as npt


def companion_matrix(ar_coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Build the companion matrix from autoregressive coefficients.

    For ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` the companion is the
    ``(kp, kp)`` matrix whose top block row is ``[A_1, ..., A_p]`` and whose
    sub-diagonal is the identity.

    Args:
        ar_coeffs: Coefficients ``A_1, ..., A_p``. Shape ``(p,)`` (scalar,
            ``k = 1``) or ``(p, k, k)`` (matrix).

    Returns:
        The companion matrix of shape ``(k * p, k * p)``.

    Raises:
        DimensionError: If ``ar_coeffs`` has an unsupported rank or is not
            square (matrix case), or if ``p < 1``.
        NumericalError: If ``ar_coeffs`` contains non-finite values.

    Example:
        >>> C = companion_matrix([0.5, -0.3])
        >>> C.shape
        (2, 2)
        >>> bool(C[1, 0] == 1.0)
        True
    """
    ar = _normalize_ar(ar_coeffs)
    p, k = ar.shape[0], ar.shape[1]
    if p < 1:
        raise DimensionError("companion_matrix requires at least one lag (p >= 1).")
    size = k * p
    companion = np.zeros((size, size), dtype=np.float64)
    companion[:k, :] = np.concatenate([ar[i] for i in range(p)], axis=1)
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return companion


def companion_from_polynomial(poly: LagPolynomial) -> npt.NDArray[np.float64]:
    """Build the companion matrix from a monic AR :class:`LagPolynomial`.

    Args:
        poly: A polynomial in monic AR form (constant term == identity).

    Returns:
        The companion matrix of shape ``(k * p, k * p)``.

    Raises:
        SpecificationError: If ``poly`` is not in monic AR form.
    """
    return companion_matrix(poly.ar_coeffs())


def selector_matrix(k: int, p: int) -> npt.NDArray[np.float64]:
    """The selector ``J = [I_k, 0, ..., 0]`` extracting the leading block.

    Used to project companion powers back to the original ``k`` variables, e.g.
    the reduced-form impulse responses ``Psi_h = J @ C**h @ J.T``.

    Args:
        k: System dimension.
        p: Lag order.

    Returns:
        A ``(k, k * p)`` array.

    Raises:
        DimensionError: If ``k < 1`` or ``p < 1``.
    """
    if k < 1 or p < 1:
        raise DimensionError(f"selector_matrix requires k >= 1 and p >= 1; got k={k}, p={p}.")
    selector = np.zeros((k, k * p), dtype=np.float64)
    selector[:, :k] = np.eye(k)
    return selector


def n_deterministic(trend: str) -> int:
    """Number of deterministic regressors implied by a trend specification.

    Args:
        trend: One of ``"n"``, ``"c"``, ``"ct"``.

    Returns:
        ``0``, ``1``, or ``2``.

    Raises:
        SpecificationError: If ``trend`` is not a recognized specification.

    Example:
        >>> n_deterministic("ct")
        2
    """
    try:
        return _TREND_WIDTH[trend]
    except KeyError:
        raise SpecificationError(
            f"trend must be one of {tuple(_TREND_WIDTH)}; got {trend!r}."
        ) from None


def deterministic_columns(trend: str, nobs: int, *, start: int = 1) -> npt.NDArray[np.float64]:
    """Build the deterministic regressor block.

    Args:
        trend: One of ``"n"``, ``"c"``, ``"ct"``.
        nobs: Number of rows.
        start: Time index of the first row, so that a conditional sample
            beginning at observation ``p + 1`` carries the trend values it
            would have had in the full sample.

    Returns:
        An ``(nobs, n_deterministic(trend))`` array.

    Example:
        >>> deterministic_columns("ct", 3, start=2)
        array([[1., 2.],
               [1., 3.],
               [1., 4.]])
    """
    width = n_deterministic(trend)
    out = np.empty((nobs, width), dtype=np.float64)
    if width >= 1:
        out[:, 0] = 1.0
    if width == 2:
        out[:, 1] = np.arange(start, start + nobs, dtype=np.float64)
    return out


def lag_matrix(
    y: npt.NDArray[np.float64], order: int, *, start: int | None = None
) -> npt.NDArray[np.float64]:
    """Build the matrix of lagged levels ``[y_{t-1}, ..., y_{t-order}]``.

    Args:
        y: The series.
        order: Number of lags; ``0`` yields a zero-width matrix.
        start: First time index retained. Defaults to ``order``, which drops
            exactly the observations that have no complete lag history.

    Returns:
        An ``(n - start, order)`` array.

    Raises:
        DimensionError: If ``start`` is smaller than ``order``, which would
            reference lags before the beginning of the series.

    Example:
        >>> lag_matrix(np.arange(5.0), 2)
        array([[1., 0.],
               [2., 1.],
               [3., 2.]])
    """
    n = y.shape[0]
    first = order if start is None else start
    if first < order:
        raise DimensionError(f"start ({first}) must be at least order ({order}).")
    if order == 0:
        return np.zeros((n - first, 0), dtype=np.float64)
    return np.column_stack([y[first - i : n - i] for i in range(1, order + 1)])


def conditional_design(
    y: npt.NDArray[np.float64], order: int, trend: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Build the target and regressor matrix for a conditional least-squares fit.

    Args:
        y: The series.
        order: Autoregressive order ``p``.
        trend: Deterministic specification.

    Returns:
        A tuple ``(target, design, nobs)`` where ``target`` is ``y[p:]``,
        ``design`` stacks the deterministic block ahead of the lag block, and
        ``nobs`` is the effective sample size ``n - p``.

    Raises:
        DimensionError: If the series is not longer than ``order``.
    """
    n = y.shape[0]
    if n <= order:
        raise DimensionError(
            f"series of length {n} is too short for a conditional design of order {order}."
        )
    eff = n - order
    det = deterministic_columns(trend, eff, start=order + 1)
    lags = lag_matrix(y, order)
    return y[order:], np.column_stack([det, lags]), eff


def psd_sqrt(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A matrix square root valid for symmetric positive-semidefinite input.

    Negative eigenvalues arising from round-off are clipped to zero, so a
    covariance that is singular or marginally indefinite still yields a usable
    factor rather than a NaN.

    Args:
        matrix: A symmetric positive-semidefinite matrix.

    Returns:
        A factor ``S`` with ``S @ S.T`` equal to ``matrix``.

    Raises:
        DimensionError: If ``matrix`` is not square.
        NumericalError: If the eigendecomposition is not finite.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"psd_sqrt requires a square matrix; got {matrix.shape}.")
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigvals)):
        raise NumericalError("psd_sqrt eigendecomposition produced non-finite values.")
    root = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
    return np.asarray(root, dtype=np.float64)


def _as_coefficient_stack(ar_coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return AR coefficients as a ``(p, k, k)`` float array (scalar -> k=1)."""
    ar = np.asarray(ar_coeffs, dtype=np.float64)
    if ar.ndim == 1:
        ar = ar.reshape(ar.shape[0], 1, 1)
    elif ar.ndim == 3:
        if ar.shape[1] != ar.shape[2]:
            raise DimensionError(
                f"Matrix AR coefficients must be square; got trailing dimensions "
                f"{ar.shape[1]}x{ar.shape[2]}."
            )
    else:
        raise DimensionError(
            f"AR coefficients must be rank 1 (scalar) or rank 3 (matrix); got rank {ar.ndim}."
        )
    if not np.all(np.isfinite(ar)):
        raise NumericalError("AR coefficients contain non-finite values.")
    return ar
