import torch
from pathlib import Path
from exp.exp_basic import Exp_basic

class Exp_audio_classification(Exp_basic):
    def __init__(self, args, device):
        # the basice class has initialized the self.args
        super().__init__(args)
        self.device = device