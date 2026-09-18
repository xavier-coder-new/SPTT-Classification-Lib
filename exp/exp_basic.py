import torch
import os
from models import BpttGRU, BpttLSTM, SpttGRU, SpttLSTM, SpttGRU_End, SpttLSTM_End, SpttLSTM_Streaming


class Exp_basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'BpttLSTM': BpttLSTM,
            'BpttGRU': BpttGRU,
            'SpttLSTM': SpttLSTM,
            'SpttGRU': SpttGRU,
            'SpttLSTM_End': SpttLSTM_End,
            'SpttGRU_End': SpttGRU_End,
            'SpttLSTM_Streaming': SpttLSTM_Streaming
        }
        self.device = self._acquire_device()
        # self.model = self._build_model().to(self.device)
        
    def _build_model(self):
        """Subclasses must implement this method"""
        raise NotImplementedError
    
    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu_id) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu_id))
            print('Use GPU: cuda:{}'.format(self.args.gpu_id))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _get_loader(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    
    