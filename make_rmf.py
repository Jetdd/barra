"""
Author: Jet Deng
Date: 2023-11-27 13:31:58
LastEditTime: 2023-11-27 13:32:24
Description: Make RMFs
"""
import pandas as pd
from config import read_path, dominant_path, contracts_path
import numpy as np

def get_close(tp: str) -> pd.DataFrame:
    """根据品种得到Concated DataFrame of the CLOSE

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=contracts, index=datetime, values=close
    """

    _path = read_path / f"{tp}.pkl"

    df = pd.read_pickle(_path)
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

    dominant_df = pd.read_csv(dominant_path / "{}.csv".format(tp)).set_index('date')
    contracts_df = pd.read_csv(contracts_path / "{}.csv".format(tp)).rename(columns={"Unnamed: 0":'date'}).set_index('date')
    new_contracts = []
    # 遍历全部合约的序列, 去掉比主力更近的近月合约
    for _date in dominant_df.index:
        dominant_contract_num = dominant_df['dominant'].loc[_date][-4:]
        daily_contracts = contracts_df.loc[_date]
        selected_contracts = [
            con
            for con in daily_contracts.dropna()
            if int(con[-4:]) >= int(dominant_contract_num[-4:])
        ]  # 得到包含主力在内的未来所有合约

        new_contracts.append(np.sort(selected_contracts))

    template = pd.DataFrame(index=dominant_df.index, data=new_contracts)
    template.columns = [f"rmf_{i+1}" for i in range(len(template.columns))]

    return template.iloc[:-1] # 用来对齐价格数据, 价格数据暂未更新到最后一天

def make_rmf(tp: str) -> pd.DataFrame:
    """制作RMF矩阵, 将TEMPLATE填充CLOSE价格, 并做换月平滑

    Args:
        tp (str): 品种名

    Returns:
        pd.DataFrame: columns=[rmf_1, ..., rmf_10], values=smoothed_close, index=date
    """
    template = get_contracts_template(tp=tp)
    close_df = get_close(tp=tp)
    
    # Mapping close prices
    rmf_list = []
    for date, row in template.iterrows():
        row = row.dropna()
        daily_close = close_df[row].loc[date]
        rmf_list.append(daily_close.values)

    
    rmf_df = pd.DataFrame(data=rmf_list, index=template.index)
    rmf_df.columns = [f"rmf_{i+1}" for i in range(len(rmf_df.columns))]
    
    # Smoothing close prices
    
    return rmf_df

template = get_contracts_template('AG')
close_df = get_close('AG')
make_rmf('AG')

