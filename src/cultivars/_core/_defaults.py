

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))
""""""

_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
""""""

_ROW_SUM_ATOL = 1e-6
""""""

_TINY = 1e-300
""""""

_SCHEMA_VERSION = 1
""""""

_D_MAX = 0.499
""""""

_PACF_CLIP = 0.999
""""""

_DEFAULT_GARCH_ORDER: dict[str, tuple[int, int, int]] = {
    "GARCH": (1, 0, 1),
    "GJR": (1, 1, 1),
    "EGARCH": (1, 1, 1),
    "FIGARCH": (1, 0, 1),
}
""""""
