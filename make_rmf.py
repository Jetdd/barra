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

    dominant_df = pd.read_csv(dominant_path / "{}.csv".format(tp)).set_index("date")
    contracts_df = (
        pd.read_csv(contracts_path / "{}.csv".format(tp))
        .rename(columns={"Unnamed: 0": "date"})
        .set_index("date")
    )
    new_contracts = []
    # 遍历全部合约的序列, 去掉比主力更近的近月合约
    for _date in dominant_df.index:
        dominant_contract_num = dominant_df["dominant"].loc[_date][-4:]
        daily_contracts = contracts_df.loc[_date]
        selected_contracts = [
            con
            for con in daily_contracts.dropna()
            if int(con[-4:]) >= int(dominant_contract_num[-4:])
        ]  # 得到包含主力在内的未来所有合约

        new_contracts.append(np.sort(selected_contracts))

    template = pd.DataFrame(index=dominant_df.index, data=new_contracts)
    template.columns = [f"rmf_{i+1}" for i in range(len(template.columns))]

    return template.iloc[:-1]  # 用来对齐价格数据, 价格数据暂未更新到最后一天


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
    dfs = []
    for col in template.columns:
        contract_series = template[col]
        mapped_close_prices = map_close_prices(
            contract_series=contract_series, close_price_df=close_df
        )
        dfs.append(mapped_close_prices)

    rmf_df = pd.concat(dfs, axis=1)
    rmf_df = rmf_df.dropna(axis=1)  # 去除有nan值的rmf列
    rmf_df.columns = [f"rmf_{i+1}" for i in range(len(rmf_df.columns))]

    # Smoothing close prices
    # 平滑规则, 换合约前4天按照4/5前合约, 1/5后合约, ...的规律进行平滑拼接
    for col in rmf_df.columns:
        prev_1_index = np.where(template[col] != template[col].shift(-1))[
            0
        ]  # 合约进行换月切换, 下一天为新合约

        # 循环平滑
        for i in range(4):
            prev_i = prev_1_index - i  # 切换前i+1天, 新合约切换前2天也就是i=1
            outcoming_contracts = template[col][prev_i]  # 旧合约
            incoming_contracts = template[col].shift(-i - 1)[
                prev_i
            ]  # 新合约在换月前i天, index需要往前i+1天
            outcoming_prices = map_close_prices(
                contract_series=outcoming_contracts, close_price_df=close_df
            )
            incoming_prices = map_close_prices(
                contract_series=incoming_contracts, close_price_df=close_df
            )
            # Update
            rmf_df[col][prev_i] = (i + 1) / 5 * outcoming_prices + (
                4 - i
            ) / 5 * incoming_prices

    return rmf_df


def map_close_prices(
    contract_series: pd.Series, close_price_df: pd.DataFrame
) -> pd.Series:
    """UTIL FUNCTION: 按照df里存下的合约template进行close prices projections

    Args:
        contract_series (pd.Series): index=date, values=contracts
        close_price_df (pd.DataFrame): close_df obtained from get_close

    Returns:
        pd.Series: index=date, values=close
    """
    # Map each contract to its corresponding close price for each date
    mapped_close_prices = contract_series.index.to_series().map(
        lambda date: close_price_df.at[date, contract_series[date]]
        if contract_series[date] in close_price_df.columns
        else np.nan
    )

    return mapped_close_prices
