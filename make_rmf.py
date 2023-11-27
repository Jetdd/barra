"""
Author: Jet Deng
Date: 2023-11-27 13:31:58
LastEditTime: 2023-11-27 13:32:24
Description: Make RMFs
"""
import pandas as pd
from config import read_path, dominant_path, contracts_path
from my_tools import my_rq_tools
import numpy as np

def get_close(tp: str) -> pd.DataFrame:
    """根据品种得到Concated DataFrame of the CLOSE

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=contracts, index=datetime, values=close
    """

    read_path = read_path / f"{tp}.pkl"

    df = pd.read_pickle(read_path)
    df = df.reset_index()
    new_df = pd.pivot_table(
        data=df, index="date", columns="order_book_id", values="close"
    )

    return new_df

def get_contracts_template(tp: str) -> pd.DataFrame:
    """根据品种从米筐动态获得当日可交易合约列表, 通过主力合约筛选掉近月到期合约, 再将可交易合约组成RMF (BARRA COM2) 矩阵

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=[rmf_1, rmf_2, ..., rmf_10], index=date, values=contract
    """

    dominant_df = pd.read_csv(dominant_path / "f{tp}.csv")
    contracts_df = pd.read_csv(contracts_path / "f{tp}.csv")
    new_contracts = []
    # 遍历全部合约的序列, 去掉比主力更近的近月合约
    for _date in dominant_df.index:
        dominant_contract_num = dominant_df.loc[_date][-4:]
        daily_contracts = contracts_df.loc[_date.date()]
        selected_contracts = [
            con
            for con in daily_contracts.dropna()
            if int(con[-4:]) >= int(dominant_contract_num)
        ]  # 得到包含主力在内的未来所有合约

        new_contracts.append(np.sort(selected_contracts))

    template = pd.DataFrame(index=dominant_df.index, data=new_contracts)
    template.columns = [f"rmf_{i+1}" for i in range(len(template.columns))]

    return template

def make_rmf(tp: str) -> pd.DataFrame:
    """制作RMF矩阵, 将TEMPLATE填充CLOSE价格, 并做换月平滑

    Args:
        tp (str): 品种名

    Returns:
        pd.DataFrame: columns=[rmf_1, ..., rmf_10], values=smoothed_close, index=date
    """
    template = get_contracts_template(tp=tp)
    close_df = get_close(tp=tp)
    
    pass