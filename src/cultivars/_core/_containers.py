# filepath: /src/cultivars/_core/_containers.py
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

"""Reusable value objects shared by internal mechanisms and public results."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, SpecificationError
from ._loaders import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl


class Standardized(NamedTuple):
    """A standardized series together with the parameters needed to invert it.

    Attributes:
        values: The standardized data, same shape as the input.
        mean: The per-column mean that was subtracted.
        scale: The per-column standard deviation that was divided out.
    """

    values: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]

    def inverse(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map standardized values back to the original scale."""
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


class LagPolynomial:
    """A scalar or matrix lag polynomial ``c_0 + c_1 L + ... + c_p L**p``.

    The coefficients are stored as an immutable float array of shape
    ``(p + 1,)`` for a scalar polynomial or ``(p + 1, k, k)`` for a matrix
    polynomial over a ``k``-dimensional system.

    Args:
        coeffs: Coefficient array, highest index = highest lag. Shape
            ``(p + 1,)`` (scalar) or ``(p + 1, k, k)`` (matrix, square).

    Raises:
        DimensionError: If ``coeffs`` is not rank 1 or rank 3, is empty, or
            (rank 3) is not square in its trailing dimensions.
        NumericalError: If ``coeffs`` contains non-finite entries.

    Example:
        >>> phi = LagPolynomial.from_ar_coeffs([0.5, -0.3])
        >>> phi.degree
        2
        >>> phi.ar_coeffs()
        array([ 0.5, -0.3])
    """

    __slots__ = ("_c",)

    # Dunder methods
    def __init__(self, coeffs: npt.ArrayLike) -> None:
        """Initialize a lag polynomial with the given coefficients."""
        c = np.asarray(coeffs, dtype=np.float64)
        if c.ndim not in (1, 3):
            raise DimensionError(
                f"LagPolynomial coefficients must be rank 1 (scalar) or rank 3 "
                f"(matrix); got rank {c.ndim} with shape {c.shape}."
            )
        if c.shape[0] < 1:
            raise DimensionError("LagPolynomial requires at least one coefficient.")
        if c.ndim == 3 and c.shape[1] != c.shape[2]:
            raise DimensionError(
                f"Matrix lag polynomial coefficients must be square; got trailing "
                f"dimensions {c.shape[1]}x{c.shape[2]}."
            )
        if not np.all(np.isfinite(c)):
            from cultivars.exceptions import NumericalError

            raise NumericalError("LagPolynomial coefficients contain non-finite values.")
        c.setflags(write=False)
        self._c = c

    def __mul__(self, other: LagPolynomial) -> LagPolynomial:
        """Polynomial product (coefficient convolution)."""
        if not isinstance(other, LagPolynomial):
            return NotImplemented
        if self.is_matrix != other.is_matrix:
            raise DimensionError("Cannot multiply a scalar and a matrix lag polynomial.")
        if self.is_matrix and self.dim != other.dim:
            raise DimensionError(
                f"Matrix polynomials have incompatible dimensions {self.dim} and {other.dim}."
            )
        if self.is_matrix:
            k = self.dim
            out = np.zeros((self.degree + other.degree + 1, k, k), dtype=np.float64)
            for i in range(self.degree + 1):
                for j in range(other.degree + 1):
                    out[i + j] += self._c[i] @ other._c[j]
            return LagPolynomial(out)
        return LagPolynomial(np.convolve(self._c, other._c))

    def __eq__(self, other: object) -> bool:
        """Check equality of two lag polynomials."""
        if not isinstance(other, LagPolynomial):
            return NotImplemented
        return self._c.shape == other._c.shape and bool(np.array_equal(self._c, other._c))

    def __repr__(self) -> str:
        """Return a string representation of the lag polynomial."""
        return f"LagPolynomial(degree={self.degree}, dim={self.dim})"

    # Class methods
    @classmethod
    def from_ar_coeffs(cls, ar_coeffs: npt.ArrayLike) -> LagPolynomial:
        """Build the monic AR polynomial ``I - A_1 L - ... - A_p L**p``.

        Args:
            ar_coeffs: The autoregressive coefficients ``A_1, ..., A_p`` as
                they appear in ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + e_t``.
                Shape ``(p,)`` (scalar) or ``(p, k, k)`` (matrix).

        Returns:
            The corresponding :class:`LagPolynomial`.

        Raises:
            DimensionError: If ``ar_coeffs`` is not rank 1 or rank 3, or
                (rank 3) is not square.

        Example:
            >>> LagPolynomial.from_ar_coeffs([0.5, -0.3]).coefficients
            array([ 1. , -0.5,  0.3])
        """
        ar = np.asarray(ar_coeffs, dtype=np.float64)
        if ar.ndim == 1:
            c = np.empty(ar.shape[0] + 1, dtype=np.float64)
            c[0] = 1.0
            c[1:] = -ar
        elif ar.ndim == 3:
            if ar.shape[1] != ar.shape[2]:
                raise DimensionError(
                    f"Matrix AR coefficients must be square; got trailing "
                    f"dimensions {ar.shape[1]}x{ar.shape[2]}."
                )
            k = ar.shape[1]
            c = np.zeros((ar.shape[0] + 1, k, k), dtype=np.float64)
            c[0] = np.eye(k)
            c[1:] = -ar
        else:
            raise DimensionError(f"AR coefficients must be rank 1 or rank 3; got rank {ar.ndim}.")
        return cls(c)

    # Properties
    @property
    def degree(self) -> int:
        """The polynomial degree ``p`` (number of lags)."""
        return int(self._c.shape[0] - 1)

    @property
    def dim(self) -> int:
        """The system dimension ``k`` (``1`` for a scalar polynomial)."""
        return 1 if self._c.ndim == 1 else int(self._c.shape[1])

    @property
    def is_matrix(self) -> bool:
        """Whether this is a matrix (multivariate) polynomial."""
        return self._c.ndim == 3

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """A writable copy of the stored coefficient array."""
        return self._c.copy()

    # Public methods
    def ar_coeffs(self, *, tol: float = 1e-10) -> npt.NDArray[np.float64]:
        """Recover the autoregressive coefficients ``A_1, ..., A_p``.

        Requires the polynomial to be monic (``c_0 == I``), i.e. a genuine AR
        polynomial rather than an arbitrary lag polynomial.

        Args:
            tol: Absolute tolerance for the ``c_0 == I`` check.

        Returns:
            The ``A_i`` with shape ``(p,)`` (scalar) or ``(p, k, k)`` (matrix).

        Raises:
            SpecificationError: If the constant term is not the identity, so
                the polynomial is not in monic AR form.
        """
        c0 = self._c[0]
        identity = np.eye(self.dim) if self.is_matrix else np.array(1.0)
        if not np.allclose(c0, identity, atol=tol):
            raise SpecificationError(
                "ar_coeffs() requires a monic polynomial (constant term == I); "
                "this polynomial is not in autoregressive form."
            )
        return -self._c[1:].copy()

    def evaluate(self, z: complex) -> npt.NDArray[np.complex128] | complex:
        """Evaluate the polynomial at a (complex) point ``z``.

        Computes ``sum_i c_i z**i``. Evaluating at ``z = exp(-1j * omega)``
        yields the frequency-domain transfer function used in spectral methods.

        Args:
            z: The point at which to evaluate; may be complex.

        Returns:
            A complex scalar (scalar polynomial) or a ``(k, k)`` complex array
            (matrix polynomial).

        Example:
            >>> phi = LagPolynomial.from_ar_coeffs([0.5, -0.3])
            >>> round(float(phi.evaluate(1.0).real), 4)
            0.8
        """
        powers = np.power(complex(z), np.arange(self.degree + 1))
        if self.is_matrix:
            return np.tensordot(powers, self._c.astype(np.complex128), axes=(0, 0))
        return complex(np.dot(powers, self._c))

    def roots(self) -> npt.NDArray[np.complex128]:
        """Roots of a scalar polynomial ``c_0 + c_1 x + ... + c_p x**p``.

        Returns:
            The complex roots.

        Raises:
            DimensionError: If the polynomial is a matrix polynomial; use the
                companion eigenvalues in :mod:`cultivars.core.stability` instead.
        """
        if self.is_matrix:
            raise DimensionError(
                "roots() is defined only for scalar polynomials; for matrix "
                "polynomials use companion eigenvalues (cultivars.core.stability)."
            )
        return np.roots(self._c[::-1]).astype(np.complex128)


class InformationCriteria(NamedTuple):
    """Model-selection criteria computed from a fitted log-likelihood.

    Attributes:
        aic: Akaike information criterion, ``-2 * llf + 2 * k``.
        bic: Bayesian (Schwarz) criterion, ``-2 * llf + k * log(n)``.
        hqic: Hannan-Quinn criterion, ``-2 * llf + 2 * k * log(log(n))``.
    """

    aic: float
    bic: float
    hqic: float

    @classmethod
    def from_likelihood(cls, llf: float, nobs: int, n_params: int) -> InformationCriteria:
        """Compute all three information criteria from a fit summary.

        Args:
            llf: Maximized log-likelihood.
            nobs: Number of observations the likelihood was evaluated on.
            n_params: Number of free parameters, including the innovation variance.

        Returns:
            The populated :class:`InformationCriteria`.

        Raises:
            SpecificationError: If ``nobs < 3`` (``log(log(n))`` is undefined or
                negative below that) or ``n_params`` is negative.

        Example:
            >>> ic = InformationCriteria.from_likelihood(-100.0, 200, 3)
            >>> round(ic.aic, 2)
            206.0
        """
        if nobs < 3:
            raise SpecificationError(f"nobs must be >= 3 to form criteria; got {nobs}.")
        if n_params < 0:
            raise SpecificationError(f"n_params must be non-negative; got {n_params}.")
        penalty = 2.0 * n_params
        return cls(
            aic=float(-2.0 * llf + penalty),
            bic=float(-2.0 * llf + n_params * np.log(nobs)),
            hqic=float(-2.0 * llf + penalty * np.log(np.log(nobs))),
        )


class _ForwardPass(NamedTuple):
    """Everything one Kalman forward pass produces, before result assembly."""

    predicted_state: npt.NDArray[np.float64]
    predicted_state_cov: npt.NDArray[np.float64]
    filtered_state: npt.NDArray[np.float64]
    filtered_state_cov: npt.NDArray[np.float64]
    loglik_contrib: npt.NDArray[np.float64]
    obs_index: list[npt.NDArray[np.intp]]
    innovation: list[npt.NDArray[np.float64]]
    innovation_precision: list[npt.NDArray[np.float64]]
    obs_design: list[npt.NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class SummaryTable:
    """A rendered-on-demand estimation summary.

    Holds the *structure* of a summary rather than a formatted string, so the
    same object can render as fixed-width text in a terminal, as HTML in a
    notebook, or as a dataframe for further analysis. Nothing is formatted
    until a renderer is called, and no optional dependency is imported unless
    one of the frame renderers is.

    The shape is deliberately narrow: a title, a block of scalar metadata
    rendered two pairs per line, one coefficient table, and a block of closing
    notes. That is what an estimation summary is; anything richer belongs in a
    purpose-built object rather than a general table.

    Attributes:
        title: Headline, typically the model specification.
        metadata: Label/value pairs describing the fit -- sample size, method,
            information criteria.
        columns: Column headers for the coefficient table.
        rows: Coefficient table rows. The first entry of each row is the
            parameter name; the rest align with ``columns[1:]``.
        notes: Closing lines, typically diagnostics or warnings.
    """

    title: str
    metadata: tuple[tuple[str, str], ...] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    notes: tuple[str, ...] = ()

    _WIDTH: ClassVar[int] = 80
    """Minimum character width.

    The rules widen to fit the metadata block and the coefficient table, but
    never to fit a note: a long closing remark wraps to the established width
    instead of stretching the whole summary around itself.
    """

    def __str__(self) -> str:
        """Render as fixed-width text."""
        return self.to_text()

    def __repr__(self) -> str:
        """Render as fixed-width text, so a bare expression shows the summary."""
        return self.to_text()

    def to_text(self) -> str:
        """Render the summary as fixed-width text.

        Returns:
            A multi-line string with a rule-delimited header, the metadata
            block in two columns, the coefficient table, and the notes wrapped
            to the table width.
        """
        meta = self._metadata_lines(self._WIDTH)
        table = self._table_lines(self._WIDTH) if self.rows else []
        width = max([len(line) for line in (*meta, *table)] + [len(self.title), self._WIDTH])
        rule = "=" * width
        lines: list[str] = [self.title.center(width).rstrip(), rule, *meta]
        if table:
            lines.extend(["-" * width, *table])
        lines.append(rule)
        for note in self.notes:
            lines.extend(textwrap.wrap(note, width=width) or [""])
        return "\n".join(lines)

    def _metadata_lines(self, width: int) -> list[str]:
        """Lay the metadata pairs out two per line.

        Cell widths come from the content rather than from a fixed fraction of
        ``width``, so a long specification label pushes the whole block wider
        instead of overflowing its own cell and shifting the second column.

        Args:
            width: Minimum total line width.

        Returns:
            One string per rendered line.
        """
        if not self.metadata:
            return []
        label_width = max(len(label) for label, _ in self.metadata) + 2
        value_width = max(len(value) for _, value in self.metadata) + 2
        cell = max(label_width + value_width, width // 2)
        lines: list[str] = []
        for i in range(0, len(self.metadata), 2):
            cells = [
                f"{label + ':':<{cell - value_width}}{value:>{value_width - 2}}  "
                for label, value in self.metadata[i : i + 2]
            ]
            lines.append("".join(cells).rstrip())
        return lines

    def _table_lines(self, width: int) -> list[str]:
        """Lay the coefficient table out with right-aligned numeric columns.

        Args:
            width: Total line width.

        Returns:
            One string per rendered line, header first.
        """
        name_width = max([len(self.columns[0])] + [len(row[0]) for row in self.rows]) + 2
        widths = [
            max([len(self.columns[j])] + [len(row[j]) for row in self.rows]) + 3
            for j in range(1, len(self.columns))
        ]
        header = self.columns[0].ljust(name_width) + "".join(
            head.rjust(w) for head, w in zip(self.columns[1:], widths, strict=True)
        )
        body = [
            row[0].ljust(name_width)
            + "".join(value.rjust(w) for value, w in zip(row[1:], widths, strict=True))
            for row in self.rows
        ]
        return [header.rstrip(), *(line.rstrip() for line in body)]

    def _repr_html_(self) -> str:
        """Render the summary as HTML for Jupyter.

        Returns:
            A self-contained ``<div>``; no stylesheet or JavaScript is emitted,
            so the output survives nbconvert and static HTML docs unchanged.
        """
        border = "border-bottom:1px solid #999;"
        label_style = "padding:1px 12px 1px 0;color:#555"
        value_style = "padding:1px 24px 1px 0;text-align:right"
        head_style = f"padding:2px 10px;text-align:right;{border}"
        cell_pad = "padding:1px 10px;text-align:"
        parts = [
            '<div style="font-family:ui-monospace,Menlo,monospace;font-size:0.85em">',
            f'<div style="font-weight:600;padding:2px 0;{border}">{self._escape(self.title)}</div>',
        ]
        if self.metadata:
            cells = "".join(
                f'<tr><td style="{label_style}">{self._escape(label)}</td>'
                f'<td style="{value_style}">{self._escape(value)}</td></tr>'
                for label, value in self.metadata
            )
            parts.append(f'<table style="border-collapse:collapse">{cells}</table>')
        if self.rows:
            head = "".join(
                f'<th style="{head_style}">{self._escape(col)}</th>' for col in self.columns
            )
            body = "".join(
                "<tr>"
                + "".join(
                    f'<td style="{cell_pad}{"left" if i == 0 else "right"}">'
                    f"{self._escape(value)}</td>"
                    for i, value in enumerate(row)
                )
                + "</tr>"
                for row in self.rows
            )
            parts.append(
                f'<table style="border-collapse:collapse;margin-top:4px">'
                f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            )
        for note in self.notes:
            parts.append(f'<div style="color:#555;padding-top:3px">{self._escape(note)}</div>')
        parts.append("</div>")
        return "".join(parts)

    def to_pandas(self) -> pd.DataFrame:
        """Return the coefficient table as a :class:`pandas.DataFrame`.

        The parameter names become the index and the metadata is attached to
        ``DataFrame.attrs``, so nothing in the summary is lost in translation.

        Returns:
            The coefficient table.

        Raises:
            ImportError: If pandas is not installed.
        """
        pd = require_optional("pandas")
        frame = pd.DataFrame(
            [row[1:] for row in self.rows],
            index=[row[0] for row in self.rows],
            columns=list(self.columns[1:]),
        )
        frame.index.name = self.columns[0] if self.columns else None
        frame.attrs["title"] = self.title
        frame.attrs["metadata"] = dict(self.metadata)
        frame.attrs["notes"] = list(self.notes)
        return frame

    def to_polars(self) -> pl.DataFrame:
        """Return the coefficient table as a :class:`polars.DataFrame`.

        Polars carries no index, so the parameter name is emitted as the first
        column rather than being promoted out of the table.

        Returns:
            The coefficient table.

        Raises:
            ImportError: If polars is not installed.
        """
        pl = require_optional("polars")
        columns = {name: [row[i] for row in self.rows] for i, name in enumerate(self.columns)}
        return pl.DataFrame(columns)

    @staticmethod
    def _escape(text: str) -> str:
        """Escape the three characters that can break emitted HTML.

        Args:
            text: Raw cell text.

        Returns:
            The escaped text.
        """
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
