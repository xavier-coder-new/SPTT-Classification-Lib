import re
import torch
from torchvision import datasets, transforms
from typing import Literal, Union
from torch.utils.data import Dataset
from datasets import load_dataset, Audio


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
    def __init__(
        self,
        path,
        subset="training",
        sample_rate=16000,
        config_name="v0.02",
    ):
        """
        Hugging Face datasets version of Speech Commands.

        subset:
            - "training" / "train"
            - "validation" / "val" / "vali"
            - "testing" / "test"
        """
        subset_map = {
            "training": "train",
            "train": "train",
            "validation": "validation",
            "val": "validation",
            "vali": "validation",
            "testing": "test",
            "test": "test",
        }
        if subset not in subset_map:
            raise ValueError(f"Invalid subset: {subset}")

        hf_split = subset_map[subset]

        ds = load_dataset(
            "google/speech_commands",
            config_name,
            split=hf_split,
            cache_dir=str(path),
        )

        # decode + resample on access
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

        self.dataset = ds
        self.sample_rate = sample_rate

        label_names = ds.features["label"].names
        self.label_to_index = {name: i for i, name in enumerate(label_names)}
        self.index_to_label = {i: name for i, name in enumerate(label_names)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        audio_info = item["audio"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32).unsqueeze(0)  # [1, T]

        # HF label may already be int class index
        if isinstance(item["label"], int):
            label_id = item["label"]
            label_name = self.index_to_label[label_id]
        else:
            label_name = item["label"]
            label_id = self.label_to_index[label_name]

        speaker_id = item.get("speaker_id", "unknown")
        utterance_number = item.get("utterance_id", 0)

        return {
            "waveform": waveform,
            "sample_rate": audio_info["sampling_rate"],
            "label": label_name,
            "label_id": label_id,
            "speaker_id": str(speaker_id),
            "utterance_number": int(utterance_number) if str(utterance_number).isdigit() else 0,
        }


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