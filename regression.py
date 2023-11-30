"""
Author: Jet Deng
Date: 2023-11-29 17:01:57
LastEditTime: 2023-11-30 13:56:41
Description: Cross-Sectional Regression
"""
import pandas as pd
from typing import Any
from modeling import rolling_pca
from statsmodels.formula.api import ols
from statsmodels.api import add_constant

def cross_regression(tp: str) -> pd.DataFrame:
    """
        r_t = X_t * f_t + epislon_t
        r_t: RMF returns at time t (N*1)
        X_t: factor exposure at time t (N*K)
        f_t: factor returns at time t (K*1)
        epislon_t: idiosyncratic returns at time t (N*1)
    Args:
        tp (str): product name

    Returns:
        pd.DataFrame: factor returns at each time t
    """
    