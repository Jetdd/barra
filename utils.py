'''
Author: Jet Deng
Date: 2023-11-27 14:15:20
LastEditTime: 2023-11-27 14:19:38
Description: DATA FEATCH FUNCTIONS
'''
from config import main_path
import pandas as pd

def get_close(tp: str) -> pd.DataFrame:
    """根据品种得到Concated DataFrame of the CLOSE

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=contracts, index=datetime, values=close
    """
    
    read_path = main_path / f"{tp}.pkl"
    
    df = pd.read_pickle(read_path)
    

def get_oi(tp: str) -> pd.DataFrame:
    """根据品种得到Concated DataFrame of the OI

    Args:
        tp (str): _description_

    Returns:
        pd.DataFrame: columns=contracts, index=datetime, values=oi
    """
    pass
    
