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
from regression import cross_regression
from covariance import compute_covariance_matrix, compute_specific_cov

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
        deal_na=True,
    )()  # Clean the rmf_df

    if normed_rmf.shape[1] < 3:  # if less than 3 factors, skip
        print(f"Less than 3 factors for {tp}")
        continue
    res = PCAModel(rmf_df=normed_rmf, n_components=3)()  # Apply PCA
    ev = res.explained_variance_ratio
    idx.append(tp)
    explained_variance.append(ev)


# df = pd.DataFrame(
#     explained_variance, index=idx, columns=["Shift", "Twist", "Butterfly"]
# )
# df

# tp='AG'
# lookback_window=240
# rolling_method='overlap'
# rmf_df = make_rmf_ret(tp=tp)
# reg_res = cross_regression(tp=tp, lookback_window=lookback_window, rolling_method=rolling_method)
# # cov_mat_dict = compute_covariance_matrix(factor_returns=reg_res.factor_returns, sample_period=240, newey_west=False)
# simga_cov_mat_dict = compute_specific_cov(factor_resid=reg_res.factor_resid, sample_period=240, newey_west=True)
# simga_cov_mat_dict['2021-01-04']


