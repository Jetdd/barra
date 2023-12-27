"""
Author: Jet Deng
Date: 2023-12-11 14:47:02
LastEditTime: 2023-12-22 14:04:49
Description: 每时刻测试风险因子的预测性
"""
from make_rmf import make_rmf_ret
from modeling import Preprocess, PCAModel, rolling_pca
from my_tools import my_tools
import numpy as np
import pandas as pd
from my_tools import my_factor_analysis, my_tools
from matplotlib import pyplot as plt
import pickle

# 计算风险因子
universe = my_tools.get_universe()
# grid_seach_dict = {}
# for lookback_window in [30, 60, 120, 240]:

#     grid_seach_dict[f"factors_{lookback_window}"] = {}  # key=date, values=factors
#     for tp in universe:
#         rmf_df = make_rmf_ret(tp=tp)  # Make the corresponding rmf_df
#         normed_rmf = Preprocess(
#             rmf_df=rmf_df,
#             threshold=3,
#             window=240,
#             deal_na=True,
#         )()  # Clean the rmf_df

#         if normed_rmf.shape[1] < 3:  # if less than 3 factors, skip
#             print(f"Less than 3 factors for {tp}")
#             continue
#         res = rolling_pca(
#             tp=tp, lookback_window=lookback_window, rolling_method="overlap", n_components=3
#         )
#         grid_seach_dict[f"factors_{lookback_window}"][tp] = res

# # Save the grid_search_dict
# with open("factors.pkl", "wb") as f:
#     pickle.dump(grid_seach_dict, f)

# Read the grid_search_dict
with open("factors.pkl", "rb") as f:
    grid_seach_dict = pickle.load(f)

# Make factors
temp_dict = {}
for lookback_window in grid_seach_dict.keys():
    grid_search_temp = grid_seach_dict[lookback_window]
    for tp in grid_search_temp.keys():
        idx = []
        data_list = []
        temp = grid_search_temp[tp]
        for date_ in temp.keys():
            idx.append(date_)
            data_list.append(temp[date_].factors[-1, :])

        temp_df = pd.DataFrame(
            data_list, index=idx, columns=["shift", "twist", "butterfly"]
        )
        temp_dict[tp] = temp_df

    shift_df = my_tools.my_dataframe(data_dict=temp_dict, string="shift").ffill()
    butterfly_df = my_tools.my_dataframe(
        data_dict=temp_dict, string="butterfly"
    ).ffill()
    twist_df = my_tools.my_dataframe(data_dict=temp_dict, string="twist").ffill()

    shift_df.index = shift_df.index.astype("datetime64[ns]").sort_values()
    butterfly_df.index = butterfly_df.index.astype("datetime64[ns]").sort_values()
    twist_df.index = twist_df.index.astype("datetime64[ns]").sort_values()

    # 测试
    for norm_method in ["cross_rank", "rolling_rank"]:
        for shift_num in [1, 5, 10]:
            data_dict = my_tools.my_load_data_2(need=universe, freq="day", adj=True)
            hold_ret = my_tools.my_hold_ret(data_dict=data_dict)
            fa = my_factor_analysis.FactorAnalysis(
                alpha=butterfly_df,
                hold_ret=hold_ret,
                norm_method=norm_method,
                num_groups=5,
                shift=shift_num,
            )
            fa.plot(
                suptitle=f"lookback_window={lookback_window},  factor_name: bf, norm_method: {norm_method}, shift_num: {shift_num}"
            )
            print(f"lookback_window={lookback_window}, bf, {norm_method}, shift_num={shift_num}")
            bt_res = fa.run()
            print(bt_res.stats)
            plt.show()
