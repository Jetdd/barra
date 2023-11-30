"""
Author: Jet Deng
Date: 2023-11-28 17:00:08
LastEditTime: 2023-11-29 09:56:48
Description: PCA Modeling
"""
from typing import Any
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
from make_rmf import make_rmf_ret


class Preprocess:
    """BARRA COM2 removes the outliers that are 10 larger than 10 stds
    and replace those between 3 stds to 10 stds with 3 stds
    """

    def __init__(self, rmf_df, threshold, window, **kwargs) -> None:
        self.rmf_df = rmf_df
        self.threshold = threshold
        self.window = window
        self.ret_type = kwargs.get("ret_type", "std")
        self.res = pd.DataFrame()
        self.deal_na = kwargs.get("deal_na", False)
        """If True, drop the na values in the rmf_df, the dates might note be continuous
        """

    def _deal_na(self) -> pd.DataFrame:
        """Deal with the na values in the rmf_df"""
        # First drop na values if the row is all na
        self.res.dropna(axis=0, how="all", inplace=True)

        if self.deal_na:
            # Then drop na values if the column is all na
            for col in self.res.columns:
                # If the column has more than 1/3 na values, drop it
                if self.res[col].isna().sum() >= self.res.shape[0] / 3:
                    self.res.drop(col, axis=1, inplace=True)
            self.res.dropna(axis=0, how="any", inplace=True)
        else:
            # Drop na values if the column has any nan values
            self.res.dropna(axis=1, how="any", inplace=True)

    def rolling_winsorize(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: winsorized rmf_ret, columns = [rmf_1, ..., rmf_n]
        """
        for col in self.rmf_df.columns:
            series = self.rmf_df[col].copy()

            upper = (
                series.rolling(window=self.window).mean()
                + self.threshold * series.rolling(window=self.window).std()
            )
            lower = (
                series.rolling(window=self.window).mean()
                - self.threshold * series.rolling(window=self.window).std()
            )
            if self.ret_type == "nan":
                series[series < lower] = np.nan
                series[series > upper] = np.nan
            else:
                series[series < lower] = lower
                series[series > upper] = upper
            self.res[col] = series
        return self.res

    def rolling_norm(self) -> pd.DataFrame:
        """ROLLING normalization
        Returns:
            pd.DataFrame: normed rmf_df
        """
        for col in self.res:
            df = self.res[col]
            _mean = df.rolling(window=self.window).mean()
            _std = df.rolling(window=self.window).std()
            normed_df = (df - _mean) / _std
            self.res[col] = normed_df
        return self.res

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.rolling_winsorize()
        self.rolling_norm()
        self._deal_na()  # Deal nan for modeling
        return self.res


@dataclass
class Report:
    n_components: int
    factors: np.array
    explained_variance_ratio: np.array
    factor_exposure: np.array


class PCAModel:
    """Use the RMF matrix to compute the set of variables including the factor exposures, etc."""

    def __init__(self, rmf_df: pd.DataFrame, n_components: int) -> None:
        self.rmf_df = rmf_df
        self.n_components = n_components
        self.factors = None
        self.explained_variance_ratio = None
        self.factor_exposure = None

    def apply_model(self):
        """PCA Modeling"""
        model = PCA(
            n_components=self.n_components,
        )
        self.factors = model.fit_transform(self.rmf_df)
        self.n_components = model.n_components_
        self.explained_variance_ratio = model.explained_variance_ratio_
        self.factor_exposure = model.components_

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.apply_model()
        res = Report(
            n_components=self.n_components,
            factors=self.factors,
            explained_variance_ratio=self.explained_variance_ratio,
            factor_exposure=self.factor_exposure,
        )

        return res


def rolling_pca(
    tp: str,
    lookback_window: int,
    rolling_method: str = "overlap",
    n_components: int = 3,
) -> dict:
    """Apply rolling PCA to the data with the fixed rolling window

    Args:
        tp (str): product name
        lookback_window (int): number of data points
        rolling_method (str): "overlap" or "non-overlap"
        n_components (int): number of components for PCA (default: 3)

    Returns:
        dict: result of rolling PCA
    """
    rmf_df = make_rmf_ret(tp=tp)
    normed_rmf = Preprocess(rmf_df=rmf_df, threshold=3, window=lookback_window)()
    res = {}
    if rolling_method == "overlap": # overlap: daily rolling
        for i in range(len(normed_rmf) - lookback_window + 1):
            window_data = normed_rmf.iloc[i : i + lookback_window]
            model = PCA(n_components=n_components)
            factors = model.fit_transform(window_data)
            explained_variance_ratio = model.explained_variance_ratio_
            factor_exposure = model.explained_variance_
            report = Report(
                n_components=n_components,
                factors=factors,
                explained_variance_ratio=explained_variance_ratio,
                factor_exposure=factor_exposure,
            )

            key = window_data.index[-1]  # The last date of the window
            res[key] = report  # Store the current result

    elif rolling_method == "non-overlap": # non-overlap: it depends on the window size
        for i in range(0, len(normed_rmf), lookback_window):
            window_data = normed_rmf.iloc[i : i + lookback_window]
            model = PCA(n_components=n_components)
            factors = model.fit_transform(window_data)
            explained_variance_ratio = model.explained_variance_ratio_
            factor_exposure = model.explained_variance_
            report = Report(
                n_components=n_components,
                factors=factors,
                explained_variance_ratio=explained_variance_ratio,
                factor_exposure=factor_exposure,
            )

            key = window_data.index[-1]  # The last date of the window
            res[key] = report  # Store the current result
    return res
