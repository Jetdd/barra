'''
Author: Jet Deng
Date: 2023-12-11 14:47:02
LastEditTime: 2023-12-11 17:45:16
Description: 每时刻测试风险因子的预测性
'''
from make_rmf import make_rmf_ret
from modeling import Preprocess, PCAModel, rolling_pca
from my_tools import my_tools
# 测试风险因子
universe = my_tools.get_universe()

factors = {}  # key=date, values=factors
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
    res = rolling_pca(tp=tp,
                      lookback_window=240,
                      rolling_method='overlap',
                      n_components=3)
    factors[tp] = res
