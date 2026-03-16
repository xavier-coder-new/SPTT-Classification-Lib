import torch
from pathlib import Path
from exp.exp_basic import Exp_basic
from data_provider.data_factory import Data_Factory
import time
from tools.utils import _setup_logger
from rich.console import Console
from tools.TrainProgressDisplay import TrainingProgress
import torch.nn as nn
from torch.optim.adam import Adam
import torch.optim.lr_scheduler as lr_scheduler
from tools.EarlyStop import Earlystopping
import math

class Exp_image_classification(Exp_basic):
    def __init__(self, args, device):
        # the basice class has initialized the self.args
        super().__init__(args)
        self.device = device
    
    # /home/leo/breeze/NMI/code/uoro_pytorch/datasets
    def _get_loader(self, data_name, path="./datasets"):
        if data_name == "sequential_mnist":
            path = "/home/leo/breeze/NMI/code/uoro_pytorch/datasets"
        
        data_loader = Data_Factory(path=Path(path), num_worker=self.args.num_worker)
        if data_name == "sequential_mnist":
            train_loader, vali_loader = data_loader.get_data_loader(
                data_name=self.args.data_name,
                mode="train", 
                batch_size=self.args.batch_size,
                need_vali=self.args.need_vali, 
                vali_ratio=self.args.vali_ratio,
                split_seed=self.args.seed,
                seq_mode=self.args.seq_mode, 
                permute=self.args.permute, 
                permutation=self.permutation if self.args.permute else None
            )
            
            test_loader = data_loader.get_data_loader(
                data_name=self.args.data_name,
                mode="test",
                need_vali=False,
                seq_mode=self.args.seq_mode,
                permute=self.args.permute,
                permutation=self.permutation if self.args.permute else None
            )
        return train_loader, vali_loader, test_loader
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model
    
    def _build_export_path(self) -> Path:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
        loss_path = (Path("loss_data") / self.args.model / self.args.data_name 
                     / f"seed_{self.args.seed}" / f"Epoch-{self.args.epochs}_Patience-{self.args.patience}_truncate_num-{self.args.truncate_num}")

        train_loss_path = loss_path /  f"train_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}_truncate_num-{self.args.truncate_num}_{timestamp}.csv"
        train_loss_path.parent.mkdir(parents=True, exist_ok=True)
        
        vali_loss_path = loss_path / f"vali_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}_truncate_num-{self.args.truncate_num}_{timestamp}.csv"
        vali_loss_path.parent.mkdir(parents=True, exist_ok=True)
        
        # visualization paths
        visual_train_loss = loss_path / f"train_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}_truncate_num-{self.args.truncate_num}_{timestamp}.svg"
        visual_vali_loss = loss_path / f"vali_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}_truncate_num-{self.args.truncate_num}_{timestamp}.svg"
        visual_train_loss.parent.mkdir(parents=True, exist_ok=True)
        visual_vali_loss.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = (Path("checkpoints") / self.args.model / self.args.data_name 
                           / f"seed_{self.args.seed}" / f"Epoch-{self.args.epochs}_Patience-{self.args.patience}_truncate_num-{self.args.truncate_num}" 
                           / f"checkpoint_{timestamp}.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        return train_loss_path, vali_loss_path, visual_train_loss, visual_vali_loss, checkpoint_path
    
        
    def train(self):
        self.file_logger, self.console_logger = _setup_logger(self.args.save, self.args, use_rich=self.args.use_rich)
        self.file_logger.info(f"Starting training with args: {self.args}")
        
        if self.args.permute:
            if self.args.data_name == "sequential_mnist":
                self.permutation = torch.randperm(784)
            elif self.args.data_name == "cifar10":
                # XXX: 
                self.permutation = torch.randperm(32*32*3)
            else:
                raise ValueError(f"{self.args.data_name} is not supported")
        else:
            self.permutation = None
            
        train_loader, vali_loader, test_loader = self._get_loader()
        (train_loss_path, vali_loss_path, 
         visual_train_loss, visual_vali_loss, 
         checkpoint_path) = self._build_export_path()
        
        if self.args.use_rich:
            console = Console()
            training_progress = TrainingProgress(total_epochs=self.args.epochs, console=console)
        else:
            raise ValueError("Please set use_rich to True to use rich progress bar")
        
        model = self._build_model()
        model.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6,
            last_epoch=-1,
            verbose=False,
        )
        
        epoch_count = 0
        first_train = True
        
        early_stopping = Earlystopping(patience=self.args.patience, verbose=True)
        
        for epoch in range(self.args.epochs):
            model.train()
            
            epoch_count += 1
            batch_count = 0
            iter_count = 0
            
            epoch_correct_num = 0
            epoch_total_num = 0
            
            epoch_loss = []
            
            # with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}", unit="batch") as tepoch:
            with training_progress.create_progress_bar("Training") as progress:
                task_id = progress.add_task(f"Epoch {epoch}", total=len(train_loader))
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    # batch_x: [batch_size, seq_len, feature_dim]
                    # batch_y: [batch_size]
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_size = batch_x.size(0)
                    # FIXME:
                    model.model.reset_state(batch_size)
                    assert torch.all(model.model.hx == 0) and torch.all(model.model.cx == 0), "Hidden state is not reset to zeros at the beginning of each batch"
                    
                    batch_count += 1
                    iter_count += 1
                    
                    num_chunks = 0
                    sequence_loss = 0.0
                    
                    sequence_length = batch_x.size(1)
                    actual_length = torch.full(
                        (batch_x.size(0),),
                        sequence_length,
                        dtype=torch.long,
                        device=self.device,
                    )
                    
                    if self.args.truncate_num == 1:
                        chunk_len = sequence_length
                    else:
                        if self.args.truncate_num > sequence_length:
                            raise ValueError(f"truncate_num {self.args.truncate_num} is greater than sequence length {sequence_length}")
                        chunk_len = math.ceil(sequence_length / self.args.truncate_num)
                        chunk_len = max(1, chunk_len)
                        
                    for start in range(0, sequence_length, chunk_len):
                        end = min(start + chunk_len, sequence_length)
                        chunk_x = batch_x[:, start:end, :] # [batch_size, chunk_len, feature_dim]
                        chunk_T = chunk_x.size(1)
                        
                        # # list of [batch_size, feature_dim], the length of inputs is chunk_T (time steps num)
                        inputs = [chunk_x[:, t, :] for t in range(chunk_T)] 
                        
                        chunk_actual_length = (actual_length - start).clamp(min=0, max=chunk_T)
                        
                        if first_train:
                            print(f"batch_x shape: {batch_x.shape}, chunk_x shape: {chunk_x.shape}")
                            print(f"inputs length: {len(inputs)}")
                            first_train=False
                            self.chunk_len = chunk_len
                            
                        optimizer.zero_grad()
                        if end != sequence_length:
                            # FIXME: 
                            output = model(
                                inputs=inputs,
                                actual_length=chunk_actual_length, 
                                sequence_length=chunk_T,
                            )
                            
                            loss = criterion(output, batch_y)
                        else:
                            assert end == sequence_length, f"end {end} is not equal to sequence_length {sequence_length}"
                            output_final = model(
                                inputs=inputs,
                                actual_length=chunk_actual_length, 
                                sequence_length=chunk_T,
                            )
                            loss = criterion(output_final, batch_y)
                            model.reset_logits()
                        
                        sequence_loss += loss
                        optimizer.step() 
                        
                        
                        
                    

                    
            