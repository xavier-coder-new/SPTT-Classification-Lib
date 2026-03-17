import torch
import argparse
import random
import numpy as np
from global_param import global_vars
from tools.utils import set_seed
from exp.exp_image_classification import Exp_image_classification
from exp.exp_audio_classification import Exp_audio_classification
from exp.exp_text_classification import Exp_text_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SPTT and BPTT for classification')
    
    parser.add_argument('--data_name', type=str, default="AG_NEWS", required=True, 
                        help="IMDB, sequential_mnist, cifar10, google_speech")
    parser.add_argument('--model', type=str, default="BpttLSTM", required=True,
                        help="BpttLSTM, BpttGRU, SpttLSTM, SpttGRU, SpttLSTM_End, SpttGRU_End")
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128, required=True)
    parser.add_argument('--epochs', type=int, default=100, required=True)
    parser.add_argument('--lr_rate', type=float, default=0.001, required=True, help="learning rate")
    parser.add_argument('--hidden_dim', type=int, default=512, required=True)
    parser.add_argument('--embedding_dim', type=int, default=256, required=True)
    parser.add_argument('--krank', type=int, default=6, required=True)
    parser.add_argument('--truncate_num', type=int, default=4, required=True)
    # parser.add_argument('--truncate_length', type=int, default=100, help='the length of truncated sequence')
    parser.add_argument('--seed', type=int, default=2025, required=True)
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type') 
    parser.add_argument('--max_length', type=int, default=400, required=True, help='max text length')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--need_vali', action="store_true", help='whether to use validation set')
    parser.add_argument('--vali_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--seq_mode', type=str, default='pixel', help='pixel or row')
    parser.add_argument('--permute', action="store_true", help='whether to permute the data')
    parser.add_argument('--use_rich', action="store_true", help='whether to use rich logger')
    parser.add_argument("--save", default="logs", help="Log directory")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    parser.add_argument("--feature_dim", type=int, help="feature dimension", required=True)
    parser.add_argument("--output_dim", type=int, help="output dimension", required=True)
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers for RNN model",)
    parser.add_argument("--use_multi_gpu", action="store_true", help="whether to use multiple GPUs")
    parser.add_argument("--num_worker", type=int, default=4, help="number of workers for data loading")
    args = parser.parse_args()

    global_vars.krank = args.krank
    
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu_id))
        print('Using GPU')
    
    if args.data_name in {'sequential_mnist', 'cifar10'}:
        exp = Exp_image_classification(args, args.device)
    elif args.data_name in {'audio'}:
        exp = Exp_audio_classification(args, args.device)
    elif args.data_name in {'text'}:
        exp = Exp_text_classification(args, args.device)
    else:
        raise ValueError(f"Unsupported Experiment: {args.data_name}")

    if args.truncate_num == 1:
        print(f"No truncation enabled.")
    else:
        print(f"Truncation enabled, the truncation number is {args.truncate_num}")

    print(f">>>>>>>start training : {args.model} on {args.data_name} with seed {args.seed}-Epochs-{args.epochs}-Patience-{args.patience}<<<<<")
    best_model_path, test_loader, num_chunks = exp.train()
    
    print(f">>>>>>>start testing : {args.model} on {args.data_name} with seed {args.seed}-Epochs-{args.epochs}-Patience-{args.patience}<<<<<")
    exp.test(best_model_path, test_loader, num_chunks)

    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported GPU type: {args.gpu_type}")
    
    
