# global_param.py
from dataclasses import dataclass, field
import torch

@dataclass
class GlobalConfig:
    time_block_num: int = 4
    krank: int = 10
    device: str = 'cuda:0'
    slide_window_nums: int = 4
    inherit_sptt: float = 1.0
    inherit_bptt: float = 0.0
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None: 
            cls._instance = super().__new__(cls)
        return cls._instance

global_vars = GlobalConfig()