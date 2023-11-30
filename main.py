"""
Author: Jet Deng
Date: 2023-11-29 09:56:58
LastEditTime: 2023-11-29 09:58:17
Description: Main File
"""

from make_rmf import make_rmf_ret
from modeling import Preprocess, PCAModel, rolling_pca
from my_tools import my_tools, my_plot
import pandas as pd


"""Apply PCA for each product in the universe"""
universe = my_tools.get_universe()
explained_variance = []  # Save the explained variance for each product
idx = []  # Save the product name
for tp in universe:
    rmf_df = make_rmf_ret(tp=tp)  # Make the corresponding rmf_df
    normed_rmf = Preprocess(
        rmf_df=rmf_df,
        threshold=3,
        window=240,
        deal_na=False,
    )()  # Clean the rmf_df

    if normed_rmf.shape[1] < 3:  # if less than 3 factors, skip
        print(f"Less than 3 factors for {tp}")
        continue
    res = PCAModel(rmf_df=normed_rmf, n_components=3)()  # Apply PCA
    ev = res.explained_variance_ratio
    idx.append(tp)
    explained_variance.append(ev)


df = pd.DataFrame(
    explained_variance, index=idx, columns=["Shift", "Twist", "Butterfly"]
)
df
# res = rolling_pca(lookback_window=240, rmf_df=ag, n_components=3)
# res['2023-11-24'].factor_exposure
# temp = get_contracts_template(tp='M')
# temp.isna().sum(axis=0)


rolling_res = rolling_pca(
    tp='AG', lookback_window=240, n_components=3, rolling_method="non-overlap"
)

idx = []
evr = []
for k in rolling_res.keys():
    idx.append(k)
    evr.append(rolling_res[k].explained_variance_ratio)
df = pd.DataFrame(evr, index=idx, columns=["Shift", "Twist", "Butterfly"])
my_plot.my_line_plot(df, title="Explained Variance Ratio")