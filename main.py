"""
Author: Jet Deng
Date: 2023-11-29 09:56:58
LastEditTime: 2023-11-29 09:58:17
Description: Main File
"""
from make_rmf import make_rmf_ret
from my_tools import my_plot
from modeling import Preprocess, PCAModel
import numpy as np
from sklearn.decomposition import PCA
ag = make_rmf_ret(tp="CU")
my_plot.my_line_plot(futures_df=ag, title="RMF Ret")

clean_rmf = Preprocess(rmf_df=ag, threshold=3, window=240)
normed_rmf = clean_rmf()
my_plot.my_line_plot(futures_df=normed_rmf, title="RMF Ret")

pca = PCAModel(rmf_df=normed_rmf, n_components=3)
res = pca.factors
res.shape