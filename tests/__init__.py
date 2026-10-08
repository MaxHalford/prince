from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.rinterface_lib
from rpy2.robjects import r as R

# SciPy and FactoMineR use independent BLAS/LAPACK implementations. GitHub runner
# changes have produced absolute differences up to 4.18e-6 while the relative error
# remains small. Keep NumPy's strict relative tolerance and allow only that measured
# cross-backend numerical noise.
FACTOMINER_RTOL = 1e-7
FACTOMINER_ATOL = 5e-6


def assert_allclose_to_factominer(actual, desired):
    np.testing.assert_allclose(
        actual,
        desired,
        rtol=FACTOMINER_RTOL,
        atol=FACTOMINER_ATOL,
    )


def load_df_from_R(code):
    df = R(code)
    if isinstance(df.names, rpy2.rinterface_lib.sexp.NULLType):
        return pd.DataFrame(np.array(df))
    return pd.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])
