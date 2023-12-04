"""
Author: Jet Deng
Date: 2023-12-04 10:16:03
LastEditTime: 2023-12-04 10:28:57
Description: Barra COM2 5.1 Covariance Matrix
"""
import pandas as pd
import numpy as np

def compute_covariance_matrix(
    factor_returns: pd.DataFrame, sample_period: int, lbd: float
) -> dict:
    """Compute the covariance matrix of the factor returns at time t with a sample period and a half life parameters

    Args:
        factor_returns (pd.DataFrame): index=date, columns['Shift', 'Twist', 'Butterfly']
        sample_period (int): lookback window size
        lbd (float): lambda, half life parameter

    Returns:
        dict: key=date, values=covariance matrix
    """
    cov_mat_dict = {}
    for t in range(len(factor_returns)-sample_period+1):
        idx = factor_returns.index[t+sample_period-1]
        # The sampled factor returns
        sampled_factor_returns = factor_returns.values[t:t+sample_period]
        # Initialize the covariance matrix
        cov_mat = np.ones((3, 3))
        
        # Compute the covariance matrix
        for i in range(len(cov_mat.shape[0])):
            for j in range(len(cov_mat.shape[1])):
                # To save time, we only compute the upper triangular matrix
                if j < i:
                    continue
                lbd_array = np.array([0.5 **(1/lbd) ** (sample_period - k) for k in range(sample_period)])
                f_k_bar = np.matmul(lbd_array, sampled_factor_returns[:, i]) / np.sum(lbd_array)
                f_l_bar = np.matmul(lbd_array, sampled_factor_returns[:, j]) / np.sum(lbd_array)
                raw_cov = np.matmul((sampled_factor_returns[:, i] - f_k_bar), (sampled_factor_returns[:, j] - f_l_bar))
                cov_mat[i, j] = np.matmul(lbd_array, raw_cov) / np.sum(lbd_array)
                cov_mat[j, i] = cov_mat[i, j] # Projection
        
        cov_mat_dict[idx] = cov_mat
    
    return cov_mat_dict