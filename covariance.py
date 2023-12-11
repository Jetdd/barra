"""
Author: Jet Deng
Date: 2023-12-04 10:16:03
LastEditTime: 2023-12-04 10:28:57
Description: Barra COM2 5.1 Covariance Matrix
"""
import pandas as pd
import numpy as np


def compute_covariance_matrix(
    factor_returns: pd.DataFrame,
    sample_period: int = 252,
    lbd: int = 30,
    newey_west: bool = True,
    nlag: int = 10,
) -> dict:
    """Compute the covariance matrix of the factor returns at time t with a sample period and a half life parameters

    Args:
        factor_returns (pd.DataFrame): index=date, columns['Shift', 'Twist', 'Butterfly']
        sample_period (int): lookback window size. Defaults to 540.
        lbd (int): lambda, half life parameter. Defaults to 90, which means 90 days weight becomes 0.5.
        newey_west (bool): whether to use Newey-West estimator to scale up the covariance matrix. Defaults to False.
        nlag (int): nlag parameter for Newey-West covariance matrix adjustment. Default to 10 as with Barra

    Returns:
        dict: key=date, values=covariance matrix
    """
    cov_mat_dict = {}
    for t in range(len(factor_returns) - sample_period + 1):
        idx = factor_returns.index[t + sample_period - 1]
        # The sampled factor returns
        sampled_factor_returns = factor_returns.values[t : t + sample_period]
        # Initialize the covariance matrix
        cov_mat = np.ones((factor_returns.shape[1], factor_returns.shape[1]))

        # Used for Newey-West estimator, as it needs the demeaned factor returns
        demeaned_sampled_factor_returns = sampled_factor_returns.copy()
        # Compute the covariance matrix
        for i in range(cov_mat.shape[0]):
            for j in range(cov_mat.shape[1]):
                # To save time, we only compute the upper triangular matrix
                if j < i:
                    continue
                lbd_array = np.array(
                    [
                        0.5 ** (1 / lbd) ** (sample_period - k)
                        for k in range(sample_period)
                    ]
                )
                f_k_bar = np.matmul(lbd_array, sampled_factor_returns[:, i]) / np.sum(
                    lbd_array
                )
                f_l_bar = np.matmul(lbd_array, sampled_factor_returns[:, j]) / np.sum(
                    lbd_array
                )
                raw_cov = (sampled_factor_returns[:, i] - f_k_bar) * (
                    sampled_factor_returns[:, j] - f_l_bar
                )  # Element-wise product
                cov_mat[i, j] = np.matmul(lbd_array, raw_cov) / np.sum(lbd_array)
                cov_mat[j, i] = cov_mat[i, j]  # Projection

            # Update the demeaned factor returns
            demeaned_sampled_factor_returns[:, i] -= f_k_bar

        # Newey-West scale covariance
        if newey_west:
            gamma_mat = np.zeros(cov_mat.shape)  # K * K matrix
            for i in range(nlag):
                gamma_i = compute_gamma_i(
                    nlag=i, factor_returns=demeaned_sampled_factor_returns
                )  # Demeaned factor returns
                bartlett_weight = 1 - (i + 1) / (nlag + 1)  # Bartlett weight
                temp_gamma_sum = bartlett_weight * (
                    gamma_i + gamma_i.T
                )  # To make sure the matrix is semi-pos definate
                gamma_mat += temp_gamma_sum
            gamma_mat += cov_mat  # gamma = gamma_0 + sum(gamma_i + gamma_i.T)
            cov_mat_dict[idx] = gamma_mat  # Save the result
        else:
            cov_mat_dict[idx] = cov_mat

    return cov_mat_dict


def compute_gamma_i(nlag: int, factor_returns: np.array) -> np.array:
    """Given each lag, compute the gamma_i matrix
    Gamma_i = 1 / T * sum_{t=1}^{T-i} F_{t} * F_{t+i}^T, where F_t is the factor returns at time t

    Args:
        lag (int): lag
        factor_returns (np.array): (N-nlag, K) matrix

    Returns:
        np.array: F_i K*K matrix
    """
    gamma_i = np.zeros(
        (factor_returns.shape[1], factor_returns.shape[1])
    )  # K * K matrix
    for i in range(factor_returns.shape[0] - nlag):
        f_i = factor_returns[i + nlag, :]  # Current factor returns
        f_i_lag = factor_returns[i, :]  # Lagged factor returns

        temp_gamma = f_i.reshape(-1, 1) * f_i_lag  # K * K gamma matrix
        gamma_i += temp_gamma  # Update the gamma matrix
    gamma_i /= factor_returns.shape[0]
    return gamma_i

def compute_gamma_sigma_i(nlag: int, factor_resid: np.array) -> np.array:
    """Given each lag, compute the gamma_i matrix for the specific returns covariance matrix

    Args:
        nlag (int): lag
        factor_returns (np.array): specific returns (N-nlag, K) matrix

    Returns:
        np.array: delta, N*N diagonal matrix
    """
    gamma_sigma = np.zeros((factor_resid.shape[1], factor_resid.shape[1]))  # N * N matrix
    for i in range(factor_resid.shape[0] - nlag):
        f_i = factor_resid[i + nlag, :]  # Current factor resid
        f_i_lag = factor_resid[i, :]  # Lagged factor resid
        for j in range(len(f_i)):
            diag_element = f_i[j] * f_i_lag[j]  # N * N gamma matrix
            gamma_sigma[j, j] += diag_element # Only update the diagonal elements, boost up the speed
    gamma_sigma /= factor_resid.shape[0]
    return gamma_sigma

def compute_specific_cov(
    factor_resid: pd.DataFrame,
    sample_period: int = 252,
    lbd: float = 30,
    newey_west: bool = True,
    nlag: int = 10,
):
    """Compute the specific covariance matrix of the factor returns at time t with a sample period and a half life parameters

    Args:
        factor_resid (pd.DataFrame): factor residuals. index=date, columns=rmf_i (T*N)
        sample_period (int, optional): sample period Defaults to 252.
        lbd (float, optional): half life. Defaults to 30.
        newey_west (bool, optional): Newey-West covariance adjustment. Defaults to True.
        nlag (int, optional): Newey-West serial correction. Defaults to 10.
    """
    cov_mat_dict = {}
    for t in range(len(factor_resid) - sample_period + 1):
        idx = factor_resid.index[t + sample_period - 1]
        # The sampled factor returns
        sampled_factor_resid = factor_resid.values[t : t + sample_period]
        # Initialize the covariance matrix
        cov_mat = np.zeros((factor_resid.shape[1], factor_resid.shape[1]))  # N*N
        # Used for Newey-West estimator, as it needs the demeaned factor returns
        demeaned_sampled_factor_resid = sampled_factor_resid.copy()  # N*N
        for i in range(cov_mat.shape[0]):
            mu_s = sampled_factor_resid[:, i]

            lbd_array = np.array(
                [0.5 ** (1 / lbd) ** (sample_period - k) for k in range(sample_period)]
            )
            mu_bar = np.matmul(lbd_array, mu_s) / np.sum(lbd_array)
            # Assume the residuals are independent
            cov_mat[i, i] = (
                np.matmul(lbd_array, mu_s - mu_bar) / np.sum(lbd_array)
            )  # Element-wise product

            # Update the demeaned factor returns
            demeaned_sampled_factor_resid[:, i] -= mu_bar
        # Newey-West scale covariance, similar logic as above
        if newey_west:
            gamma_mat = np.zeros(cov_mat.shape)  # N * N matrix
            for i in range(nlag):
                gamma_i = compute_gamma_sigma_i(
                    nlag=i, factor_resid=demeaned_sampled_factor_resid
                )  # Demeaned factor returns
                bartlett_weight = 1 - (i + 1) / (nlag + 1)  # Bartlett weight
                temp_gamma_sum = bartlett_weight * (
                    gamma_i + gamma_i.T
                )  # To make sure the matrix is semi-pos definate
                gamma_mat += temp_gamma_sum
            gamma_mat += cov_mat  # gamma = gamma_0 + sum(gamma_i + gamma_i.T)
            cov_mat_dict[idx] = gamma_mat  # Save the result
        else:
            cov_mat_dict[idx] = cov_mat

    return cov_mat_dict
