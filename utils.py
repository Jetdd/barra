'''
Author: Jet Deng
Date: 2023-11-27 14:15:20
LastEditTime: 2023-11-27 16:17:00
Description: DATA FEATCH FUNCTIONS
'''
from config import main_path, rq_account, rq_password
import pandas as pd
import rqdatac as rq
rq.init(rq_account, rq_password)

def get_close(tp: str) -> pd.DataFrame:
    """根据品种得到Concated DataFrame of the CLOSE

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=contracts, index=datetime, values=close
    """
    
    read_path = main_path / f"{tp}.pkl"
    
    df = pd.read_pickle(read_path)
    df = df.reset_index()
    new_df = pd.pivot_table(data=df, index='date', columns='order_book_id', values='close')
    
    return new_df


def get_template(tp: str) -> pd.DataFrame:
    """根据品种从米筐动态获得当日可交易合约列表, 通过主力合约筛选掉近月到期合约, 再将可交易合约组成RMF (BARRA COM2) 矩阵

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=[rmf_1, rmf_2, ..., rmf_10], index=date, values=contract
    """
    
    