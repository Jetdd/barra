'''
Author: Jet Deng
Date: 2023-11-27 10:37:23
LastEditTime: 2023-11-28 14:17:58
Description: Config File. Storing read paths
'''
from pathlib import Path

read_path = Path("D:/projects/data/all/daybar")

dominant_path = Path("D:/projects/barra/saved_contracts/dominant")
contracts_path = Path("D:/projects/barra/saved_contracts/all")

dominant_path.mkdir(parents=True, exist_ok=True)
contracts_path.mkdir(parents=True, exist_ok=True)