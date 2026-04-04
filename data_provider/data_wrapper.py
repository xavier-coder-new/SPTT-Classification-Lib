import re
import torch
from torchvision import datasets, transforms
from typing import Literal, Union
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Audio, Dataset as HFDataset
import torchaudio
import os
import pandas as pd

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

class SequentialCIFAR10Dataset(Dataset):
    def __init__(
        self,
        path: str,
        is_train: bool,
        need_vali: bool,
        is_download: bool,
        seq_mode: Literal["pixel", "row"] = "pixel",
        permute: bool = False,
        permutation: Union[torch.Tensor, None] = None,
        to_grayscale: bool = True,
        normalize: bool = False,
    ):
        """
        LRA-style sequential CIFAR-10 dataset.

        Args:
            path:
                dataset root
            is_train:
                whether to load training split
            need_vali:
                whether this split may later be used with random_split
            is_download:
                whether to download the dataset
            seq_mode:
                - "pixel":
                    grayscale: [1024, 1]
                    rgb      : [1024, 3]
                - "row":
                    grayscale: [32, 32]
                    rgb      : [32, 96]   # flatten channel dim into feature dim
            permute:
                whether to apply a fixed permutation on spatial positions
            permutation:
                permutation over 1024 spatial positions
            to_grayscale:
                True -> convert RGB to grayscale first (recommended for LRA-like setup)
            normalize:
                whether to normalize to roughly zero mean / unit variance
        """
        self.path = path
        self.is_train = is_train
        self.need_vali = need_vali
        self.is_download = is_download
        self.seq_mode = seq_mode
        self.permute = permute
        self.to_grayscale = to_grayscale
        self.normalize = normalize

        transform_list = [transforms.ToTensor()]  # [C, 32, 32], in [0,1]

        if self.to_grayscale:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

        if self.normalize:
            if self.to_grayscale:
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
            else:
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        self.cifar10 = datasets.CIFAR10(
            root=self.path,
            train=self.is_train,
            download=self.is_download,
            transform=transforms.Compose(transform_list),
        )

        seq_len = 32 * 32  # 1024 spatial positions

        if self.need_vali and self.permute:
            if permutation is None:
                raise ValueError("Permutation must be provided when need_vali is True")
            if permutation.numel() != seq_len:
                raise ValueError("Permutation must have 1024 elements")
            self.permutation = permutation

        elif (not self.need_vali) and self.permute:
            if permutation is None:
                self.permutation = torch.randperm(seq_len)
            else:
                if permutation.numel() != seq_len:
                    raise ValueError("Permutation must have 1024 elements")
                self.permutation = permutation
        else:
            self.permutation = None

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        x, y = self.cifar10[idx]
        # x:
        #   grayscale -> [1, 32, 32]
        #   rgb       -> [3, 32, 32]

        if self.seq_mode == "pixel":
            if self.to_grayscale:
                # [1, 32, 32] -> [1024, 1]
                x = x.view(-1)
                if self.permutation is not None:
                    x = x[self.permutation]
                x = x.unsqueeze(-1)
            else:
                # [3, 32, 32] -> [1024, 3]
                x = x.permute(1, 2, 0).reshape(-1, 3)
                if self.permutation is not None:
                    x = x[self.permutation]

        elif self.seq_mode == "row":
            if self.to_grayscale:
                # [1, 32, 32] -> [32, 32]
                x = x.squeeze(0)
                if self.permutation is not None:
                    x = x.reshape(-1)[self.permutation].reshape(32, 32)
            else:
                # [3, 32, 32] -> [32, 96]
                x = x.permute(1, 2, 0).reshape(32, 32 * 3)
                if self.permutation is not None:
                    # permutation is defined over 1024 spatial positions,
                    # so we first reshape to [1024, 3], permute, then reshape back
                    x_spatial = x.reshape(32 * 32, 3)[self.permutation]
                    x = x_spatial.reshape(32, 32 * 3)
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


class ESC50Dataset(Dataset):
    def __init__(
        self,
        path,
        mode="train",
        sample_rate=16000,
        fold=None,
        offline_first=True,
    ):
        """
        Hugging Face version of ESC-50 using: ashraq/esc50

        Args:
            path:
                HF cache root directory, e.g. "./datasets"
            mode:
                - "train" / "training"
                - "val" / "vali" / "validation"
                - "test" / "testing"
            sample_rate:
                target sampling rate
            fold:
                ESC-50 uses 5 folds. We use:
                    test fold = fold
                    val fold  = ((fold % 5) + 1)
                    train     = remaining 3 folds
                Default fold = 1
            offline_first:
                first try reusing local HF cache
        """
        self.path = path
        self.sample_rate = sample_rate
        self.fold = 1 if fold is None else int(fold)
        self.offline_first = offline_first

        if self.fold not in {1, 2, 3, 4, 5}:
            raise ValueError("fold must be one of {1,2,3,4,5}")

        mode = mode.lower()
        if mode in {"train", "training"}:
            self.mode = "train"
        elif mode in {"val", "vali", "validation"}:
            self.mode = "validation"
        elif mode in {"test", "testing"}:
            self.mode = "test"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.dataset = self._load_esc50_dataset()

        required_cols = {"filename", "fold", "target", "category", "audio"}
        missing_cols = required_cols - set(self.dataset.column_names)
        if missing_cols:
            raise ValueError(f"Missing required columns in ashraq/esc50: {missing_cols}")

        test_fold = self.fold
        val_fold = (self.fold % 5) + 1

        if self.mode == "test":
            self.dataset = self.dataset.filter(lambda x: x["fold"] == test_fold)
        elif self.mode == "validation":
            self.dataset = self.dataset.filter(lambda x: x["fold"] == val_fold)
        else:
            self.dataset = self.dataset.filter(
                lambda x: (x["fold"] != test_fold) and (x["fold"] != val_fold)
            )

        label_names = sorted(set(self.dataset["category"]))
        self.label_to_index = {name: i for i, name in enumerate(label_names)}
        self.index_to_label = {i: name for i, name in enumerate(label_names)}

        # 让 audio 列在访问时自动重采样到目标采样率
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    def _load_esc50_dataset(self):
        if self.offline_first:
            try:
                print(f"[ESC50] Trying offline HF cache from: {self.path}")
                dataset = load_dataset(
                    "ashraq/esc50",
                    split="train",
                    cache_dir=str(self.path),
                    download_mode="reuse_dataset_if_exists",
                )
                print("[ESC50] Loaded from existing HF cache or local processed cache.")
                return dataset
            except Exception as e:
                print(f"[ESC50] Offline-first cache load failed: {e}")
                print("[ESC50] Falling back to normal load_dataset(...).")

        dataset = load_dataset(
            "ashraq/esc50",
            split="train",
            cache_dir=str(self.path),
        )
        print("[ESC50] Loaded with normal load_dataset().")
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        audio_info = item["audio"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        elif waveform.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # 双保险：如果有多通道，压成单通道
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        label_name = str(item["category"])
        label_id = int(item["target"])

        return {
            "waveform": waveform,                         # [1, T]
            "sample_rate": int(audio_info["sampling_rate"]),
            "label": label_name,
            "label_id": label_id,                        # 0..49
            "filename": item["filename"],
            "fold": int(item["fold"]),
        }
        
class NSynthDataset(Dataset):
    def __init__(
        self,
        path,
        mode="train",
        sample_rate=16000,
        label_type="family",   # "family" | "source" | "instrument"
        config_name="full",
        offline_first=True,
    ):
        """
        Hugging Face / TFDS-style NSynth loader.

        NSynth official fixed splits:
            - train
            - valid
            - test

        Recommended label_type:
            - "family": 11-way classification
            - "source": 3-way classification
            - "instrument": 1006-way classification

        Args:
            path:
                HF cache root directory, e.g. "./datasets"
            mode:
                - "train" / "training"
                - "val" / "vali" / "validation" / "valid"
                - "test" / "testing"
            sample_rate:
                target sampling rate
            label_type:
                classification target
            config_name:
                NSynth config name, default "full"
            offline_first:
                first try reusing local HF cache
        """
        self.path = path
        self.sample_rate = sample_rate
        self.label_type = str(label_type).lower()
        self.config_name = config_name
        self.offline_first = offline_first

        mode = mode.lower()
        if mode in {"train", "training"}:
            self.split = "train"
            self.mode = "train"
        elif mode in {"val", "vali", "validation", "valid"}:
            self.split = "valid"
            self.mode = "validation"
        elif mode in {"test", "testing"}:
            self.split = "test"
            self.mode = "test"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if self.label_type not in {"family", "source", "instrument"}:
            raise ValueError("label_type must be one of {'family', 'source', 'instrument'}")

        self.dataset = self._load_nsynth_dataset()

        required_cols = {"audio", "id", "instrument", "pitch", "velocity"}
        missing_cols = required_cols - set(self.dataset.column_names)
        if missing_cols:
            raise ValueError(f"Missing required columns in NSynth dataset: {missing_cols}")

        # make audio auto-resample on access
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

        self._build_label_metadata()

    def _load_nsynth_dataset(self):
        if self.offline_first:
            try:
                print(f"[NSYNTH] Trying offline HF cache from: {self.path}")
                dataset = load_dataset(
                    "jg583/NSynth",
                    self.config_name,
                    split=self.split,
                    cache_dir=str(self.path),
                    download_mode="reuse_dataset_if_exists",
                    trust_remote_code=True,
                )
                print("[NSYNTH] Loaded from existing HF cache or local processed cache.")
                return dataset
            except Exception as e:
                print(f"[NSYNTH] Offline-first cache load failed: {e}")
                print("[NSYNTH] Falling back to normal load_dataset(...).")

        dataset = load_dataset(
            "jg583/NSynth",
            self.config_name,
            split=self.split,
            cache_dir=str(self.path),
            trust_remote_code=True,
        )
        print("[NSYNTH] Loaded with normal load_dataset().")
        return dataset

    def _build_label_metadata(self):
        """
        Build label name/id mappings for the selected label granularity.
        """
        # fixed mappings from NSynth dataset card / TFDS
        family_id_to_name = {
            0: "bass",
            1: "brass",
            2: "flute",
            3: "guitar",
            4: "keyboard",
            5: "mallet",
            6: "organ",
            7: "reed",
            8: "string",
            9: "synth_lead",
            10: "vocal",
        }

        source_id_to_name = {
            0: "acoustic",
            1: "electronic",
            2: "synthetic",
        }

        if self.label_type == "family":
            self.index_to_label = family_id_to_name
            self.label_to_index = {v: k for k, v in family_id_to_name.items()}
        elif self.label_type == "source":
            self.index_to_label = source_id_to_name
            self.label_to_index = {v: k for k, v in source_id_to_name.items()}
        else:
            # instrument label is a fine-grained class id (0..1005)
            # Names are not guaranteed to be human-readable in the HF sample itself,
            # so here we expose them as instrument_<id>.
            instrument_ids = sorted({int(x["label"]) for x in self.dataset["instrument"]})
            self.index_to_label = {idx: f"instrument_{idx}" for idx in instrument_ids}
            self.label_to_index = {v: k for k, v in self.index_to_label.items()}

    def __len__(self):
        return len(self.dataset)

    def _extract_label(self, item):
        instrument_info = item["instrument"]

        if self.label_type == "family":
            label_id = int(instrument_info["family"])
            label_name = self.index_to_label[label_id]
        elif self.label_type == "source":
            label_id = int(instrument_info["source"])
            label_name = self.index_to_label[label_id]
        else:
            label_id = int(instrument_info["label"])
            label_name = self.index_to_label[label_id]

        return label_name, label_id

    def __getitem__(self, index):
        item = self.dataset[index]

        audio_info = item["audio"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        elif waveform.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # safety: collapse multi-channel if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        label_name, label_id = self._extract_label(item)

        instrument_info = item["instrument"]

        return {
            "waveform": waveform,                               # [1, T]
            "sample_rate": int(audio_info["sampling_rate"]),
            "label": label_name,
            "label_id": label_id,

            # extra metadata
            "nsynth_id": item["id"],
            "pitch": int(item["pitch"]),
            "velocity": int(item["velocity"]),
            "instrument_family_id": int(instrument_info["family"]),
            "instrument_source_id": int(instrument_info["source"]),
            "instrument_label_id": int(instrument_info["label"]),
        }

class IMDBDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        max_length=None,
        offline_first=True,
    ):
        """
        IMDB dataset loader.

        Args:
            path: HF cache root directory, e.g. "./datasets"
            split: "train" / "test" / "unsupervised"
            max_length: truncate token list length
            offline_first:
                - True: first try to load from existing HF cache
                - False: directly allow normal online/cached loading
        """
        self.path = path
        self.split = split
        self.max_length = max_length
        self.offline_first = offline_first

        self.label_to_index = {0: 0, 1: 1}
        self.index_to_label = {0: "neg", 1: "pos"}

        self.dataset = self._load_imdb_dataset()

    def _load_imdb_dataset(self):
        # Option 1: Prioritize strict offline reading of HF cache
        if self.offline_first:
            try:
                print(f"[IMDB] Trying offline HF cache from: {self.path}")
                dataset = load_dataset(
                    "imdb",
                    split=self.split,
                    cache_dir=str(self.path),
                    download_mode="reuse_dataset_if_exists",
                )
                print("[IMDB] Loaded from existing HF cache or local processed cache.")
                return dataset
            except Exception as e:
                print(f"[IMDB] Offline-first cache load failed: {e}")
                print("[IMDB] Falling back to normal load_dataset(...).")

        # Option 2: Normal loading (Reuse if there is cache, download if not)
        dataset = load_dataset(
            "imdb",
            split=self.split,
            cache_dir=str(self.path),
        )
        print("[IMDB] Loaded with normal load_dataset().")
        return dataset

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
    def __init__(
        self,
        path,
        split="train",
        max_length=None,
        offline_first=True,
    ):
        """
        AG_NEWS dataset loader.

        Args:
            path: HF cache root directory, e.g. "./datasets"
            split: "train" / "test"
            max_length: truncate token list length
            offline_first:
                - True: first try to load from existing HF cache
                - False: directly allow normal online/cached loading
        """
        self.path = path
        self.split = split
        self.max_length = max_length
        self.offline_first = offline_first

        self.index_to_label = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

        self.dataset = self._load_ag_news_dataset()

    def _load_ag_news_dataset(self):
        # Option 1: Prioritize strict offline reading of HF cache
        if self.offline_first:
            try:
                print(f"[AG_NEWS] Trying offline HF cache from: {self.path}")
                dataset = load_dataset(
                    "ag_news",
                    split=self.split,
                    cache_dir=str(self.path),
                    download_mode="reuse_dataset_if_exists",
                )
                print("[AG_NEWS] Loaded from existing HF cache or local processed cache.")
                return dataset
            except Exception as e:
                print(f"[AG_NEWS] Offline-first cache load failed: {e}")
                print("[AG_NEWS] Falling back to normal load_dataset(...).")

        # Option 2: Normal loading (Reuse if there is cache, download if not)
        dataset = load_dataset(
            "ag_news",
            split=self.split,
            cache_dir=str(self.path),
        )
        print("[AG_NEWS] Loaded with normal load_dataset().")
        return dataset

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
        
class ByteIMDBDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        max_length=4000,
        offline_first=True,
        encoding="utf-8",
        add_eos=False,
    ):
        """
        LRA-style byte-level IMDB dataset.

        Args:
            path:
                HF cache root directory, e.g. "./datasets"
            split:
                "train" / "test"
            max_length:
                target byte sequence length before collate-time padding/truncation logic
                for LRA IMDB, common length is around 4000
            offline_first:
                first try reusing HF cache
            encoding:
                text encoding, default utf-8
            add_eos:
                whether to append one EOS byte token
        """
        self.path = path
        self.split = split
        self.max_length = max_length
        self.offline_first = offline_first
        self.encoding = encoding
        self.add_eos = add_eos

        self.label_to_index = {0: 0, 1: 1}
        self.index_to_label = {0: "neg", 1: "pos"}

        # byte vocabulary:
        # pad_id = 0
        # actual bytes 0..255 are shifted to 1..256
        self.pad_idx = 0
        self.byte_vocab_size = 257  # 0 for PAD, 1..256 for bytes

        # optional EOS token
        self.eos_idx = 257 if add_eos else None
        if add_eos:
            self.byte_vocab_size = 258

        self.dataset = self._load_imdb_dataset()

    def _load_imdb_dataset(self):
        if self.offline_first:
            try:
                print(f"[BYTE_IMDB] Trying offline HF cache from: {self.path}")
                dataset = load_dataset(
                    "imdb",
                    split=self.split,
                    cache_dir=str(self.path),
                    download_mode="reuse_dataset_if_exists",
                )
                print("[BYTE_IMDB] Loaded from existing HF cache or local processed cache.")
                return dataset
            except Exception as e:
                print(f"[BYTE_IMDB] Offline-first cache load failed: {e}")
                print("[BYTE_IMDB] Falling back to normal load_dataset(...).")

        dataset = load_dataset(
            "imdb",
            split=self.split,
            cache_dir=str(self.path),
        )
        print("[BYTE_IMDB] Loaded with normal load_dataset().")
        return dataset

    def __len__(self):
        return len(self.dataset)

    def _text_to_byte_ids(self, text: str):
        raw_bytes = text.encode(self.encoding, errors="replace")
        # shift by +1 so that 0 can be reserved for PAD
        byte_ids = [b + 1 for b in raw_bytes]

        if self.add_eos:
            byte_ids.append(self.eos_idx)

        if self.max_length is not None:
            byte_ids = byte_ids[:self.max_length]

        return byte_ids

    def __getitem__(self, index):
        item = self.dataset[index]
        text = item["text"]
        label = int(item["label"])

        byte_ids = self._text_to_byte_ids(text)

        return {
            "text": text,
            "byte_ids": byte_ids,
            "length": len(byte_ids),
            "label": self.index_to_label[label],
            "label_id": self.label_to_index[label],
        }