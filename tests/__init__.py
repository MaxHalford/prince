from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.rinterface_lib
from rpy2.robjects import r as R


def load_df_from_R(code):
    df = R(code)
    if isinstance(df.names, rpy2.rinterface_lib.sexp.NULLType):
        return pd.DataFrame(np.array(df))
    return pd.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])
