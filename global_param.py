import torch
class global_vars:
    time_backward = 0.0
    total_time_block_size = 0.0
    compression_finished = False
    time_block_num = 4
    krank = 10
    device = 'cuda:0'
    flag = True
    length_vocab = 0
    pad_idx = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_vars.device = device
