# global_param.py
from dataclasses import dataclass, field
import torch

@dataclass
class GlobalConfig:
    time_backward: float = 0.0
    total_time_block_size: float = 0.0
    compression_finished: bool = False
    time_block_num: int = 4
    krank: int = 10
    device: str = 'cuda:0'
    flag: bool = True
    length_vocab: int = 0
    pad_idx: int = None
    
    # 类级别的单例
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 创建全局实例
global_vars = GlobalConfig()