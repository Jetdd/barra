"""
Author: Jet Deng
Date: 2023-11-29 09:56:58
LastEditTime: 2023-11-29 09:58:17
Description: Main File
"""
from make_rmf import make_rmf_ret
from my_tools import my_plot
from modeling import preprocess, pca_model
import numpy as np
ag = make_rmf_ret(tp="CU")
my_plot.my_line_plot(futures_df=ag, title="RMF Ret")

clean_rmf = preprocess(rmf_df=ag, threshold=3, window=240)
normed_rmf = clean_rmf()
my_plot.my_line_plot(futures_df=normed_rmf, title="RMF Ret")


model = pca_model(normed_rmf)
np.sum(model.explained_variance_)
for i, each in enumerate(model.explained_variance_):
    print(f"{i}: {each/np.sum(model.explained_variance_)}")