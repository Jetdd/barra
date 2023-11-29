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
        """ROLLING NORM MATRIX
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
        return self.res.dropna()


@dataclass
class Report:
    n_components: int
    factors: np.array
    explained_variance_ratio: np.array


class PCAModel:
    def __init__(self, rmf_df: pd.DataFrame, n_components: int) -> None:
        self.rmf_df = rmf_df
        self.n_components = n_components
        self.factors = None
        self.explained_variance_ratio = None

    def _model(self):
        """PCA Modeling"""
        model = PCA(
            n_components=self.n_components,
        )
        self.factors = model.fit_transform(self.rmf_df)
        self.n_components = model.n_components_
        self.explained_variance_ratio = model.explained_variance_ratio_

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._model()
        res = Report(
            n_components=self.n_components,
            factors=self.factors,
            explained_variance_ratio=self.explained_variance_ratio,
        )

        return res
