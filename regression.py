"""
Author: Jet Deng
Date: 2023-11-29 17:01:57
LastEditTime: 2023-11-30 13:56:41
Description: Cross-Sectional Regression
"""
import pandas as pd
from typing import Any
from modeling import rolling_pca
import statsmodels.api as sm
from make_rmf import make_rmf_ret
from dataclasses import dataclass
import numpy as np

@dataclass
class Report:
    factor_returns: pd.DataFrame
    factor_exposure: dict
    factor_resid: pd.DataFrame


def cross_regression(
    tp: str, lookback_window: int, rolling_method: str
) -> pd.DataFrame:
    """
        r_t = X_t * f_t + epislon_t
        r_t: RMF returns at time t (N*1)
        X_t: factor exposure at time t (N*K)
        f_t: factor returns at time t (K*1)
        epislon_t: idiosyncratic returns at time t (N*1)
    Args:
        tp (str): product name
        lookback_window (int): rolling window size for PCA
        rolling_method (str): rolling method for PCA, "overlap" or "non-overlap"

    Returns:
        pd.DataFrame: factor returns at each time t
    """
    rmf_ret = make_rmf_ret(tp=tp)
    rolling_pca_res = rolling_pca(
        tp=tp, lookback_window=lookback_window, rolling_method=rolling_method
    )
    factor_returns = []
    resid = []
    idx = []
    exposure = {}
    for t in rolling_pca_res.keys():
        # Get the factor exposure at time t
        X_t = rolling_pca_res[t].factor_exposure
        # Get the RMF returns at time t
        r_t = rmf_ret.loc[t, :]
        # Add constant to the factor exposure
        X_t = sm.add_constant(X_t.T)
        # Fit the regression
        model = sm.OLS(r_t, X_t).fit()
        # Get the factor returns at time t
        f_t = model.params[1:].to_list()
        # Save the factor returns and residuals
        idx.append(t)
        factor_returns.append(f_t)
        resid.append(model.resid.to_list())
        exposure[t] = X_t
        
    factor_returns = pd.DataFrame(
        factor_returns, index=idx, columns=["Shift", "Twist", "Butterfly"]
    )
    resid = pd.DataFrame(
        resid, index=idx, columns=[f"resid_{i+1}" for i in range(len(resid[-1]))]
    )
    
    return Report(factor_returns=factor_returns, factor_exposure=exposure, factor_resid=resid)


def compute_r_square(tp: str, factor_resid: pd.DataFrame) -> pd.Series:
    """Compute the r-square of the cross-sectional regression given each date

    Args:
        tp (str): 品种名

    Returns:
        pd.Series: r-square
    """
    rmf_df = make_rmf_ret(tp=tp)

    aligned_rmf_df = rmf_df.loc[factor_resid.index, :] # 对齐index
    r_square = 1 - np.square(factor_resid).sum(axis=1) / np.square(aligned_rmf_df).sum(axis=1)
    return r_square

if __name__ == "__main__":
    tp = "AG"
    lookback_window = 240
    rolling_method = "overlap"
    res = cross_regression(
        tp=tp, lookback_window=lookback_window, rolling_method=rolling_method
    )
    
    temp_res = compute_r_square(tp=tp, factor_resid=res.factor_resid)
    temp_res.quantile(q=0.01)
    res.factor_returns