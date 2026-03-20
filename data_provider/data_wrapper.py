import torch
from torchvision import datasets, transforms
from typing import Literal
from torch.utils.data import Dataset
import torchaudio
from typing import Union
from torchtext.datasets import IMDB, AG_NEWS
from torchtext.data.utils import get_tokenizer

class SequentialMNISTDataset(Dataset):
    def __init__(self, path:str, is_train: bool, need_vali: bool, 
                 is_download: bool, seq_mode: Literal["pixel", "row"], 
                 permute:bool, permutation: Union[torch.Tensor, None] = None):
        """
        Load dataset for Sequential MNIST
        
        Args:
            path: path to the data
            is_train: whether to load train or test data
            is_download: whether to download the data if not present
            seq_mode: whether to load data in pixel-wise or row-wise manner
                -  "pixel": [784, 1]
                -  "row": [28, 28]
            permute: whether to permute the data
                - True: permute the data according to the given permutation
                - False: do not permute the data, kkeping the original order [28, 28]
            permutation: the permutation to apply if permute is True
                - None: use the default permutation --> torch.randperm(784), Randomly generate a random permutation from 0 to 783
        
        Note:
            If seq_mode is "pixel", then the data is loaded in pixel-wise manner.
            If seq_mode is "row", then the data is loaded in row-wise manner.
            If you want to creat the validation set, you need to split the train data into train and validation set.
            
        """
        
        self.path = path
        self.is_train = is_train
        self.need_vali = need_vali
        self.is_download = is_download
        self.seq_mode = seq_mode
        self.permute = permute
        
        self.mnist = datasets.MNIST(
            root=self.path,
            train=self.is_train,
            download=self.is_download,
            transform=transforms.ToTensor(), # Convert the PIL Image to a tensor, [1, 28, 28], and normalize the pixel values to [0, 1]
        )
        
        
        if self.need_vali and self.permute:
            if permutation is None:
                raise ValueError("Permutation must be provided when need_vali is True")
            else:
                if permutation.numel() != 784:
                    raise ValueError("Permutation must have 784 elements")
                else:
                    self.permutation = permutation
                
        if not self.need_vali and self.permute:
            if permutation is None:
                self.permutation = torch.randperm(784) # Randomly generate a random permutation from 0 to 783
            else:
                if permutation.numel() != 784:
                    raise ValueError("Permutation must have 784 elements")
                self.permutation = permutation
                
        if not self.permute:
            self.permutation = None

        
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """Return the sequential data and label at the given index
        
        Args:
            idx: index of the sample
            
        Returns:
            x: Sequential data
            - pixel mode: Tensor[784, 1], each timestep has 1 feature
            - row mode: Tensor[28, 28], each timestep has 28 features (one row)
            y: Class label (int in 0-9)
        """
        x, y = self.mnist[idx]  # x: [1, 28, 28], y: int -> MNIST only has one channel.
        
        if self.seq_mode == "pixel":
            # --> [784, 1]
            x = x.view(-1) # [784]]
            if self.permutation is not None:
                x = x[self.permutation]
            x = x.unsqueeze(-1) # [784, 1]
            
        elif self.seq_mode == "row":
            # --> [28, 28]
            x = x.squeeze(0) # [28, 28]
            if self.permutation is not None:
                x = x.reshape(-1)[self.permutation].reshape(28, 28)
        else:
            raise ValueError("seq_mode must be 'pixel' or 'row'")
        
        return x, y

class GoogleSpeechDataset(Dataset):
    def __init__(self, path, subset="training", download=False, url="speech_commands_v0.02"):
        """
        Load dataset for Google Speech Commands
        
        Dataset Introduction:
            - A single piece of data can usually be understood as
                - waveform.shape == [1, T]
                - 1 is the number of channels
                - T represents the number of sampling points in terms of time length
                - If the audio is approximately 1 second long and the sampling rate is 16 KHZ, the common ones are: [1, 16000]
            - for train
                - x = sample["waveform"], y = sample["label_id"]
                - "sample_rate", "speaker_id", "utterance_number" are only auxiliary information and are usually not directly fed into the classification model.
                
            - Original data: [batch_size, 1, T_max]
                - Solution One: if you want to directly feed the data into RNNs, you need to reshape the data to [batch_size, T_max, 1]. 
                But the sequence length is different for each sample, so you need to pad the data to the same length.
                And maybe the sequence length is too long.
                - Solution Two: You can use the MelSpectrogram provided by the torchaudio official to perform feature extraction transformation, 
                converting the original audio into a mel-time-frequency representation.
                    - [batch_size, channels, n_mels, time_frames]
                    - convert for RNNs: [batch_size, time_frames, channels * n_mels]
                
                
        
        
        Args:
            path: path to the data
            subset: which subset to load
                - "training": training data
                - "validation": validation data
                - "testing": testing data
            download: whether to download the data if not present
            url: version of the data to download, the latest version is "speech_commands_v0.02".
        """
        self.google_speech = torchaudio.datasets.SPEECHCOMMANDS(
            root=path,
            download=download,
            subset=subset,
            url=url,
        )
        
        # extract the labels and create a mapping from label to index, label --> index
        
        """
        self.google_speech = [
            (audio1, speaker1, "cat",   extra_info1...),
            (audio2, speaker2, "dog",   extra_info2...),
            (audio3, speaker3, "cat",   extra_info3...),  # 重复的标签
            (audio4, speaker4, "bird",  extra_info4...),
        ]
        
        label: {"cat", "dog", "bird"} 
        sorted_labels: ["bird", "cat", "dog"] # according to the order of the letter
        
        label_to_index: 
            {
                "bird": 0, 
                "cat": 1, 
                "dog": 2,
            }
        function:
            Convert label to index, and as the input/output of the model.
            
        index_to_label: 
            {
                0: "bird", 
                1: "cat", 
                2: "dog",
            }
            
        function:
            Convert the predicted index to the label and calculate the loss.
        """
        
        
        labels = sorted({label for _, _, label, *_ in self.google_speech})
        self.label_to_index = {label: i for i, label in enumerate(labels)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.google_speech)
    
    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.google_speech[index]
        # waveform: [channel, time] -> default: [1, num_samples]
        
        label_id = self.label_to_index[label]
        
        return {
            "waveform": waveform,                   # Tensor, [1, num_samples]
            "sample_rate": sample_rate,             # int
            "label": label,                         # str
            "label_id": label_id,                   # int
            "speaker_id": speaker_id,               # str
            "utterance_number": utterance_number,   # int
        }


class IMDBDataset(Dataset):
    def __init__(self, path, split="train", tokenizer_name="basic_english", max_length=None):
        """
        Load dataset for IMDB sentiment classification.
        
        Args:
            path: dataset cache path
            split: "train" or "test"
            tokenizer_name: tokenizer used by torchtext
            max_length: truncate token length if provided.
            
        Returns each sample:
            {
                "text": original text,
                "tokens": list[str],
                "label": str,       # "pos" or "neg"
                "label_id": int,    # 0 or 1
            }
        """
        
        self.path = path
        self.split = split
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.max_length = max_length
        
        # torchtext.datasets.IMDB returns iterable samples: (label, text)
        self.samples = list(IMDB(root=self.path, split=self.split))
        
        self.label_to_index = {"neg": 0, "pos": 1}
        self.index_to_label = {0: "neg", 1: "pos"}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        label, text = self.samples[index]
        tokens = self.tokenizer(text)
        
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
            
        label_id = self.label_to_index[label]
        
        return {
            "text": text,
            "tokens": tokens,
            "label": label,
            "label_id": label_id
        }

class AGNewsDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        tokenizer_name="basic_english",
        max_length=None,
    ):
        """
        Load dataset for AG_NEWS text classification

        Args:
            path: dataset cache path
            split: "train" or "test"
            tokenizer_name: tokenizer used by torchtext
            max_length: truncate token length if provided

        Returns each sample:
            {
                "text": original text,
                "tokens": list[str],
                "label": int,         # 1,2,3,4 from torchtext
                "label_id": int,      # 0,1,2,3
            }
        """
        self.path = path
        self.split = split
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.max_length = max_length

        # torchtext AG_NEWS returns iterable samples: (label, text)
        self.samples = list(AG_NEWS(root=self.path, split=self.split))

        # 原始标签是 1~4，这里转成 0~3
        self.label_to_index = {
            1: 0,  # World
            2: 1,  # Sports
            3: 2,  # Business
            4: 3,  # Sci/Tech
        }
        self.index_to_label = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, text = self.samples[index]
        tokens = self.tokenizer(text)

        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        label_id = self.label_to_index[label]

        return {
            "text": text,
            "tokens": tokens,
            "label": label,
            "label_id": label_id,
        }