import logging
import argparse
from pathlib import Path
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def _setup_logger(save_dir: str, args:argparse.Namespace, use_rich: bool = True):
    # Pa_Ep:{args.patience}_{args.max}_LR_{args.rate}_T_{args.truncate}_off_{str(args.off)}
    # full_log_path = (Path(save_dir) / args.exp_type / args.model / args.data_name / str(args.seed) / f"Epoch-{args.epochs}_Patience-{args.patience}"
    #                  / f"krank_{args.krank}_Trun_num_{args.truncate_num}_Slide_{args.slide_window_nums}" )
    
    if args.use_clip:
        clip_norm = args.clip_norm
    else:
        clip_norm = "None"
        
    if args.bptt_low_rank:
        low_rank = f"_Bptt_LowRank_Using"
    else:
        low_rank = "_Bptt_LowRank_None"
    
    
    full_log_path = (Path(save_dir) / args.exp_type / args.model / args.data_name / f'clip_norm_{clip_norm}' / str(args.seed) / f"Epoch-{args.epochs}_Patience-{args.patience}"
                     / f"{low_rank}_krank_{args.krank}_Trun_num_{args.truncate_num}_Slide_{args.slide_window_nums}" )
    
    full_log_path.mkdir(parents=True, exist_ok=True)
    
    file_logger = logging.getLogger(f"{args.model}_{args.data_name}_seed{args.seed}__FILE")
    file_logger.setLevel(logging.INFO)
    
    console_logger = logging.getLogger(f"{args.model}_{args.data_name}_seed{args.seed}__CONSOLE")
    console_logger.setLevel(logging.INFO)
    
    if file_logger.handlers:
        file_logger.handlers.clear()
    
    if console_logger.handlers:
        console_logger.handlers.clear()
        
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_filename = (
        f"train_log-Pa_{args.patience}_Epo_{args.epochs}_Krank_{args.krank}_"
        f"LR-{args.lr_rate}_Trun_num-{args.truncate_num}_Slide_{args.slide_window_nums}-{timestamp}.log"
    )
    log_file_path = full_log_path / log_filename
    
    file_handler = logging.FileHandler(
        log_file_path,
        mode='a',
        encoding='utf-8'
    )
    
    file_format = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    file_logger.addHandler(file_handler)
    # Prevent logs from being propagated upward to the root logger to avoid duplication
    file_logger.propagate = False 
    
    if not use_rich:
        # console handler (options)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        # if you want to print to console, please uncomment the following line
        # console_logger.addHandler(console_handler)
        console_logger.propagate = False
        
    return file_logger, console_logger

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def visual_loss(loss, label='loss', name='./pic/loss.pdf'):
    plt.figure()
    plt.plot(loss, label=label, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel(label)
    plt.title(f"{label} Curve (per batch)")
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    
def calculate_accuracy(preds, labels):
    output = torch.argmax(preds, dim=1) # [batch_size]
    correct = (output == labels).sum().item()
    total = labels.size(0)
    return correct, total