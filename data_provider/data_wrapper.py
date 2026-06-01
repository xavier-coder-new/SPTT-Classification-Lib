import re
import torch
from torchvision import datasets, transforms
from typing import Literal, Union
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Audio, Dataset as HFDataset
import torchaudio
import os
import pandas as pd
from pathlib import Path
def basic_english_tokenizer(text: str):
    """
    A lightweight tokenizer similar to basic_english.
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9'.,!?;:()\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def listops_tokenizer(text: str):
    """
    Tokenizer for LRA Long ListOps.
    Keep operators / brackets / parentheses / digits as separate tokens.
    Example:
        "( ( [MAX 2 9 ] ) 4 )"
    """
    # delete whitespace at the beginning and end of the text
    text = text.strip()
    # extract [MIN [MAX [MED [SM ] ( ) and number
    tokens = re.findall(r"\[MIN|\[MAX|\[MED|\[SM|\]|\(|\)|\d+", text)
    return tokens


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
        
class LongListOpsDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        max_length=2000,
        offline_first=True,
        task_name="basic",
    ):
        """
        Long ListOps dataset loader for LRA-style TSV files.

        Expected TSV columns:
            - Source: expression string
            - Target: label (0~9)

        Common file names:
            basic_train.tsv / basic_val.tsv / basic_test.tsv
            train.tsv / val.tsv / test.tsv
            train.csv / val.csv / test.csv   # if user converted formats manually
        """
        self.path = Path(path) / "long_listops"
        self.split = split.lower()
        self.max_length = max_length
        self.offline_first = offline_first
        self.task_name = task_name

        if self.split in {"validation", "vali"}:
            self.split = "val"
        elif self.split in {"testing"}:
            self.split = "test"

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        # ListOps labels are 0~9
        self.label_to_index = {i: i for i in range(10)}
        self.index_to_label = {i: str(i) for i in range(10)}

        self.data = self._load_listops_data()

    def _resolve_file_path(self):
        candidates = [
            os.path.join(self.path, f"{self.task_name}_{self.split}.tsv"),
            os.path.join(self.path, f"{self.split}.tsv"),
            os.path.join(self.path, f"{self.task_name}_{self.split}.csv"),
            os.path.join(self.path, f"{self.split}.csv"),
        ]

        for fp in candidates:
            if os.path.exists(fp):
                return fp

        raise FileNotFoundError(
            f"Cannot find Long ListOps split file for split='{self.split}' under: {self.path}\n"
            f"Tried: {candidates}"
        )

    def _load_listops_data(self):
        file_path = self._resolve_file_path()

        sep = "\t" if file_path.endswith(".tsv") else ","
        df = pd.read_csv(file_path, sep=sep)

        # 兼容大小写
        rename_map = {}
        for col in df.columns:
            low = col.lower()
            if low == "source":
                rename_map[col] = "Source"
            elif low == "target":
                rename_map[col] = "Target"

        df = df.rename(columns=rename_map)

        if "Source" not in df.columns or "Target" not in df.columns:
            raise ValueError(
                f"Long ListOps file must contain 'Source' and 'Target' columns, got: {list(df.columns)}"
            )

        records = []
        for _, row in df.iterrows():
            text = str(row["Source"]).strip()
            label_id = int(row["Target"])

            tokens = listops_tokenizer(text)
            if self.max_length is not None:
                tokens = tokens[:self.max_length]

            records.append({
                "text": text,
                "tokens": tokens,
                "label": self.index_to_label[label_id],
                "label_id": label_id,
            })

        return records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
        