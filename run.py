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
                        help="IMDB, sequential_mnist, cifar10, google_speech, esc50, AG_NEWS, byte_imdb, nsynth")
    parser.add_argument('--model', type=str, default="BpttLSTM", required=True,
                        help="BpttLSTM, BpttGRU, SpttLSTM, SpttGRU, SpttLSTM_End, SpttGRU_End")
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128, required=True)
    parser.add_argument('--epochs', type=int, default=100, required=True)
    parser.add_argument('--lr_rate', type=float, default=0.001, required=True, help="learning rate")
    parser.add_argument('--hidden_dim', type=int, default=512, required=True)
    parser.add_argument('--embed_dim', type=int, default=256, required=True)
    parser.add_argument('--krank', type=int, default=6, required=True)
    parser.add_argument('--truncate_num', type=int, default=4, required=True)
    # parser.add_argument('--truncate_length', type=int, default=100, help='the length of truncated sequence')
    parser.add_argument('--seed', type=int, default=2025, required=True)
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type') 
    parser.add_argument('--max_length', type=int, default=None, help='max text length')
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
    parser.add_argument("--slide_window_nums", type=int, default=4, help="number of sliding windows for SPTT")
    parser.add_argument("--min_freq", type=int, default=1, help="minimum frequency for vocabulary")
    parser.add_argument("--fixed_length", type=int, default=None, help="fixed length for text")
    parser.add_argument("--length_mode", type=str, default="pad", help="mode for handling text length, including pad and repeat")
    parser.add_argument("--input_type", type=str, default="feature", help="input type, including text or feature")
    parser.add_argument("--vocab_size", type=int, default=None, help="vocabulary size")
    parser.add_argument("--pad_idx", type=int, default=0, help="padding index for text")
    parser.add_argument("--exp_type", type=str, default="text", help="experiment type, including text, image and audio", required=True)
    parser.add_argument("--offline", action="store_true", help="whether to use offline data loading")
    parser.add_argument("--n_mels", type=int, default=64, help="number of mel filters")
    parser.add_argument("--sample_rate", type=int, default=16000, help="audio sample rate")
    parser.add_argument("--esc50_fold", type=int, default=1, help="ESC50 fold number for testing, default is 1")
    parser.add_argument("--n_fft", type=int, default=2048, help="feedforward network dimension for transformer model")
    parser.add_argument("--hop_length", type=int, default=512, help="hop length for STFT")
    parser.add_argument("--to_grayscale", action="store_true", help="whether to convert image to grayscale")
    parser.add_argument("--normalize", action="store_true", help="whether to normalize the data")
    parser.add_argument("--nsynth_label_type", type=str, default="family", help="label type for NSynth dataset, including family, instrument, source")
    parser.add_argument("--nsynth_config", type=str, default="full", help="configuration for NSynth dataset")

    args = parser.parse_args()

    global_vars.krank = args.krank
    
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    
    if args.exp_type == "text":
        args.input_type = "text"
        print(f"Input type: {args.input_type}")
    elif args.exp_type in {"image", "audio"}:
        args.input_type = "feature"
        print(f"Input type: {args.input_type}")
    else:
        raise ValueError(f"Unsupported experiment type: {args.exp_type}")
    
    
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu_id))
        print('Using GPU')
    
    if args.truncate_num > 1:
        if args.slide_window_nums > 1:
            raise ValueError("Slide window need to set to 1 when truncation is enabled")
    
    if args.data_name in {'sequential_mnist', 'cifar10'}:
        exp = Exp_image_classification(args, args.device)
    elif args.data_name in {'google_speech', 'esc50', 'nsynth'}:
        exp = Exp_audio_classification(args, args.device)
    elif args.data_name in {'imdb', 'ag_news', 'byte_imdb'}:
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
    
    
