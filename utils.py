'''
Author: Jet Deng
Date: 2023-11-27 17:45:42
LastEditTime: 2023-11-28 09:45:49
Description: UTILS
'''
from config import dominant_path, contracts_path
from my_tools import my_rq_tools
from my_tools import my_tools
def get_contracts(tp: str) -> None:
    """保存主力合约列表和所有当日可交易合约列表

    Args:
        tp (str): 品种名
    """
    
    dominant_df = my_rq_tools.get_dominant(tp=tp)  # 获取主力序列
    contracts_df = my_rq_tools.get_contracts(tp=tp).dropna(
        axis=0, how="all"
    )  # 得到当日所有可交易的合约

    dominant_df.to_csv(dominant_path / f"{tp}.csv")
    contracts_df.to_csv(contracts_path / f"{tp}.csv")
    
    print(f"{tp} saved complete")
    

if __name__ == "__main__":
    universe = my_tools.get_universe()
    for tp in universe:
        get_contracts(tp=tp)