'''
Author: Jet Deng
Date: 2023-11-27 10:37:23
LastEditTime: 2023-11-27 17:45:08
Description: Config File. Storing read paths
'''
from pathlib import Path

read_path = Path("D:/projects/data/daybar/unadj")

dominant_path = Path.mkdir("D:/projects/barra/saved_contracts/dominant", exist_ok=True)
contracts_path = Path.mkdir("D:/projects/barra/saved_contracts/all", exist_ok=True)