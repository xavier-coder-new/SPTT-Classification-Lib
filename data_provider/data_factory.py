import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_provider import data_wrapper
from torch.utils.data import DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from icecream import ic

class Data_Factory:
    def __init__(self, path: str, num_worker=4, sample_rate=16000, n_mels=64) -> None:
        self.path = path
        self.num_worker = num_worker
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            # The audio signal is converted to 128-dimensional Mel spectral characteristics at each time point.
            n_mels=n_mels,
        )
        
    def speech_commands_collate_fn(self, batch):
        """
        Prepare batch for RNN/GRU/LSTM with pack_padded_sequence.

        Input batch:
            batch = [
                {"waveform": [1, T1], "sample_rate": int, "label": str, "label_id": int, ...},
                {"waveform": [1, T2], "sample_rate": int, "label": str, "label_id": int, ...},
                ...
            ]

        Returns:
            {
                "features": [B, T_max, n_mels],   # RNN input
                "lengths": [B],                   # valid mel time frames for each sample
                "label_id": [B],
                "label": list[str],
                "sample_rate": [B],
                "speaker_id": list[str],
                "utterance_number": [B],
            }
        """
        feature_list = []
        lengths = []
        labels = []
        sample_rates = []
        label_names = []
        speaker_ids = []
        utterance_numbers = []
        
        for item in batch:
            waveform = item["waveform"]  # [1, T]
            
            # Apply MelSpectrogram transformation to singel waveform
            # T: The number of sampling points of the original audio
            # time_frames: The number of frames cut out along the time axis after being transformed into spectral features
            mel = self.mel_transform(waveform) # [1, T] -> [1, n_mels, time_frames]
            mel = mel.squeeze(0).transpose(0, 1) # [time_frames, n_mels]
            feature_list.append(mel)
            lengths.append(mel.size(0))
            labels.append(item["label_id"])
            sample_rates.append(item["sample_rate"])
            label_names.append(item["label"])
            speaker_ids.append(item["speaker_id"])
            utterance_numbers.append(item["utterance_number"])

        # Pad the features to the same length
        padded_features = pad_sequence(feature_list, batch_first=True)  # [B, T_max, n_mels]

        return {
            "features": padded_features, # [B, T_max, n_mels]
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "sample_rate": torch.tensor(sample_rates, dtype=torch.long),
            "speaker_id": speaker_ids,
            "utterance_number": torch.tensor(utterance_numbers, dtype=torch.long),
        }

    def get_data_loader(self, data_name: str, mode: str, batch_size=128, path=None, 
                        need_vali=True, vali_ratio=0.1, split_seed=2025, seq_mode="pixel", 
                        download=True, permute=True, permutation: Union[torch.Tensor, None] = None
                        ) -> torch.utils.data.DataLoader:
        """
        Get data loader for the specified data name
        
        Args:
            data_name: name of the data
            path: path to the data
            mode: {"train", "test"}
            need_vali: whether to get validation set
            seq_mode: pixel or row (only applicable for sequential_mnist)
            download: whether to download the data if not present
            permute: whether to permute the data
            permutation: permutation tensor
        
        Returns:
            data loader
        """
        data_name = data_name.lower()
        mode = mode.lower()
        
        if path is None:
            path = self.path # use the default path if not provided
            
        if data_name == "sequential_mnist":
            if mode in {"train", "training"}:
                is_train = True
                shuffle = True
            elif mode in {"test", "testing"}:
                is_train = False
                shuffle = False
                ic(f"id_train:{is_train}")
            else:
                raise ValueError("Invalid mode")
            
            dataset = data_wrapper.SequentialMNISTDataset(
                path=path,
                is_train=is_train,
                need_vali=need_vali,
                is_download=download,
                seq_mode=seq_mode, # "pixel" -> [784,1], "row" -> [28,28]
                permute=permute,
                permutation=permutation
            )
            
            if need_vali:
                train_size = int((1 - vali_ratio) * len(dataset)) # 54000
                vali_size = len(dataset) - train_size # 6000
                generator  = torch.Generator().manual_seed(split_seed)
                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset, 
                    [train_size, vali_size],
                    generator=generator,
                )
                
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=True,
                    num_workers=self.num_worker,
                    pin_memory=torch.cuda.is_available(),
                )
                
                vali_loader = DataLoader(
                    dataset=vali_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=self.num_worker,
                    pin_memory=torch.cuda.is_available(),
                )
                return train_loader, vali_loader
            else:
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=True,
                    num_workers=self.num_worker,
                    pin_memory=torch.cuda.is_available(), 
                )
            
                return loader
               
        elif data_name == "google_speech":
            # Implementation for Google Speech loader
            if mode in {"train", "training"}:
                subset = "training"
                shuffle = True
            elif mode in {"val", "validation", "vali"}:
                subset = "validation"
                shuffle = False
            elif mode in {"test", "testing"}:
                subset = "testing"
                shuffle = False
            else:
                raise ValueError("Invalid mode")
            
            dataset = data_wrapper.GoogleSpeechDataset(
                path=path,
                subset=subset,
                download=download,
                url="speech_commands_v0.02",
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_worker,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.speech_commands_collate_fn,
            )
            
            return loader
        
        else:
            raise ValueError("Invalid data name")
    
