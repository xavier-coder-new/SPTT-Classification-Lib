import torch
import argparse
import random
import numpy as np
from global_param import global_vars
from tools.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SPTT and BPTT for classification')
    
    parser.add_argument('--data_name', type=str, default="AG_NEWS", required=True, 
                        help="IMDB, AG_NEWS, YelpReviewFull, DBpedia, SogouNews, AmazonReviewFull, YahooAnswers, YelpReviewPolarity")
    parser.add_argument('--model', type=str, default="BpttLSTM", required=True,
                        help="BpttLSTM, BpttGRU, SpttLSTM, SpttGRU, SpttLSTM_End, SpttGRU_End")
    parser.add_argument('--use_gpu', type=bool, default=True, required=True)
    parser.add_argument('--gpu_id', type=int, default=0, required=True)
    parser.add_argument('--batch_size', type=int, default=128, required=True)
    parser.add_argument('--itr', type=int, default=1, required=True, help="the number of experiment")
    parser.add_argument('--epochs', type=int, default=100, required=True)
    parser.add_argument('--lr_rate', type=float, default=0.001, required=True, help="learning rate")
    parser.add_argument('--hidden_dim', type=int, default=512, required=True)
    parser.add_argument('--embedding_dim', type=int, default=256, required=True)
    parser.add_argument('--krank', type=int, default=10, required=True)
    parser.add_argument('--truncate_num', type=int, default=4, required=True)
    parser.add_argument('--truncate_length', type=int, default=100, help='the length of truncated sequence')
    parser.add_argument('--is_training', type=int, default=1, help="status")
    parser.add_argument('--seed', type=int, default=2025, required=True)
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type') 
    parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate \
                            type1:阶段衰减, type2:余弦退火')
    parser.add_argument('--max_length', type=int, default=400, required=True, help='max text length')
    parser.add_argument("--off", action="store_true", help="If set, use TBPTT only")
    parser.add_argument('--classification_num', type=int, default=4, help='classification number')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--need_vali', action="store_true", help='whether to use validation set')
    parser.add_argument('--vali_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--seq_mode', type=str, default='pixel', help='pixel or row')
    parser.add_argument('--permute', action="store_true", help='whether to permute the data')
    parser.add_argument('--use_rich', action="store_true", help='whether to use rich logger')
    parser.add_argument("--save", default="logs", help="Log directory")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")

    args = parser.parse_args()

    global_vars.krank = args.krank
    
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu_id))
        print('Using GPU')
    
    print('Arg in experiment')
    

    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()
    
    
