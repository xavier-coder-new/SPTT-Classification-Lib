import re
import torch
from torchvision import datasets, transforms
from typing import Literal, Union
from torch.utils.data import Dataset
from datasets import load_dataset, Audio, Dataset as HFDataset
import torchaudio

def basic_english_tokenizer(text: str):
    """
    A lightweight tokenizer similar to basic_english.
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9'.,!?;:()\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


class SequentialMNISTDataset(Dataset):
    def __init__(self, path:str, is_train: bool, need_vali: bool,
                 is_download: bool, seq_mode: Literal["pixel", "row"],
                 permute:bool, permutation: Union[torch.Tensor, None] = None):
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
            transform=transforms.ToTensor(),
        )

        if self.need_vali and self.permute:
            if permutation is None:
                raise ValueError("Permutation must be provided when need_vali is True")
            if permutation.numel() != 784:
                raise ValueError("Permutation must have 784 elements")
            self.permutation = permutation

        elif (not self.need_vali) and self.permute:
            if permutation is None:
                self.permutation = torch.randperm(784)
            else:
                if permutation.numel() != 784:
                    raise ValueError("Permutation must have 784 elements")
                self.permutation = permutation
        else:
            self.permutation = None

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        x, y = self.mnist[idx]

        if self.seq_mode == "pixel":
            x = x.view(-1)
            if self.permutation is not None:
                x = x[self.permutation]
            x = x.unsqueeze(-1)   # [784, 1]

        elif self.seq_mode == "row":
            x = x.squeeze(0)      # [28, 28]
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


# class GoogleSpeechDataset(Dataset):
#     def __init__(
#         self,
#         path,
#         subset="training",
#         sample_rate=16000,
#         config_name="v0.02",
#     ):
#         """
#         Hugging Face datasets version of Speech Commands.

#         subset:
#             - "training" / "train"
#             - "validation" / "val" / "vali"
#             - "testing" / "test"
#         """
#         subset_map = {
#             "training": "train",
#             "train": "train",
#             "validation": "validation",
#             "val": "validation",
#             "vali": "validation",
#             "testing": "test",
#             "test": "test",
#         }
#         if subset not in subset_map:
#             raise ValueError(f"Invalid subset: {subset}")

#         # 不要自己拼了，直接判断 subset 并使用你刚刚复制出来的绝对路径
#         if subset in ["training", "train"]:
#             # 把下面这串红色的字符串，替换成你刚刚右键复制出来的真实路径！
#             arrow_file_path = "/mnt/hard_disk/weihao/SPTT-Classification-Lib/datasets/speech_commands/v0.02/0.2.0/ba3d9a6cf49aa1313c51abe16b59203451482ccb9fee6d23c94fecabf3e206da/speech_commands-train.arrow" 
            
#         elif subset in ["validation", "val", "vali"]:
#             # 同样替换为 validation.arrow 的绝对路径
#             arrow_file_path = "/mnt/hard_disk/weihao/SPTT-Classification-Lib/datasets/speech_commands/v0.02/0.2.0/ba3d9a6cf49aa1313c51abe16b59203451482ccb9fee6d23c94fecabf3e206da/speech_commands-validation.arrow"
            
#         elif subset in ["testing", "test"]:
#             # 同样替换为 test.arrow 的绝对路径
#             arrow_file_path = "/mnt/hard_disk/weihao/SPTT-Classification-Lib/datasets/speech_commands/v0.02/0.2.0/ba3d9a6cf49aa1313c51abe16b59203451482ccb9fee6d23c94fecabf3e206da/speech_commands-test.arrow"

#         print(f"正在纯离线加载音频数据: {arrow_file_path}")

#         # ds = load_dataset(
#         #     "speech_commands",
#         #     config_name,
#         #     split=hf_split,
#         #     cache_dir="/mnt/hard_disk/datasets",
#         # )
        
#         ds = HFDataset.from_file(arrow_file_path)

#         # decode + resample on access
#         ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

#         self.dataset = ds
#         self.sample_rate = sample_rate

#         if "label" in ds.features and hasattr(ds.features["label"], "names"):
#             label_names = ds.features["label"].names
#         else:
#             # 万一离线文件里没存标签名，我们手动备用写死 Google Speech V2 的 35 个类
#             print("Google Speech V2 35 classes:")
#             label_names = ["backward", "bed", "bird", "cat", "dog", 
#                            "down", "eight", "five", "follow", "forward", 
#                            "four", "go", "happy", "house", "learn", "left", 
#                            "marvin", "nine", "no", "off", "on", "one", "right", 
#                            "seven", "sheila", "six", "stop", "three", "tree", 
#                            "two", "up", "visual", "wow", "yes", "zero"]
#         self.label_to_index = {name: i for i, name in enumerate(label_names)}
#         self.index_to_label = {i: name for i, name in enumerate(label_names)}

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         item = self.dataset[index]

#         audio_info = item["audio"]
#         waveform = torch.tensor(audio_info["array"], dtype=torch.float32).unsqueeze(0)  # [1, T]

#         # HF label may already be int class index
#         if isinstance(item["label"], int):
#             label_id = item["label"]
#             label_name = self.index_to_label[label_id]
#         else:
#             label_name = item["label"]
#             label_id = self.label_to_index[label_name]

#         speaker_id = item.get("speaker_id", "unknown")
#         utterance_number = item.get("utterance_id", 0)

#         return {
#             "waveform": waveform,
#             "sample_rate": audio_info["sampling_rate"],
#             "label": label_name,
#             "label_id": label_id,
#             "speaker_id": str(speaker_id),
#             "utterance_number": int(utterance_number) if str(utterance_number).isdigit() else 0,
#         }


class IMDBDataset(Dataset):
    def __init__(self, path, split="train", max_length=None):
        """
        Hugging Face datasets version of IMDB.
        label: 0=neg, 1=pos
        """
        self.path = path
        self.split = split
        self.max_length = max_length

        self.dataset = load_dataset(
            "imdb",
            split=split,
            cache_dir=str(path),
        )

        self.label_to_index = {0: 0, 1: 1}
        self.index_to_label = {0: "neg", 1: "pos"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        text = item["text"]
        label = int(item["label"])
        tokens = basic_english_tokenizer(text)

        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        return {
            "text": text,
            "tokens": tokens,
            "label": self.index_to_label[label],
            "label_id": self.label_to_index[label],
        }


class AGNewsDataset(Dataset):
    def __init__(self, path, split="train", max_length=None):
        """
        Hugging Face datasets version of AG_NEWS.
        HF label is usually 0..3 already.
        """
        self.path = path
        self.split = split
        self.max_length = max_length

        self.dataset = load_dataset(
            "ag_news",
            split=split,
            cache_dir=str(path),
        )

        self.index_to_label = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        text = item["text"]
        label_id = int(item["label"])
        tokens = basic_english_tokenizer(text)

        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        return {
            "text": text,
            "tokens": tokens,
            "label": self.index_to_label[label_id],
            "label_id": label_id,
        }