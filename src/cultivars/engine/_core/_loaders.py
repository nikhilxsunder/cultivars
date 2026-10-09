# filepath: /src/cultivars/_core/_loaders.py
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

from importlib import import_module
from types import ModuleType

from ._defaults import _EXTRA


def require_optional(module: str) -> ModuleType:
    """Import an optional frame backend, or explain how to install it.

    Args:
        module: Top-level module name, ``"pandas"`` or ``"polars"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed, naming the extra that
            provides it.

    Example:
        >>> require_optional("numpy").__name__
        'numpy'
    """
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        extra = _EXTRA.get(module, module)
        raise ImportError(
            f"{module} is required for this conversion but is not installed; "
            f"install it with `pip install cultivars[{extra}]`."
        ) from exc
