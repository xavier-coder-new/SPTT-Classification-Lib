import torch
import os
import sys
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_provider import data_wrapper
from torch.utils.data import DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from icecream import ic


class SimpleVocab:
    def __init__(self, stoi, itos, unk_token="<unk>"):
        self.stoi = stoi
        self.itos = itos
        self.unk_token = unk_token
        self.default_index = stoi[unk_token]
        
    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi[token]

    def set_default_index(self, idx):
        self.default_index = idx

    def lookup_indices(self, tokens):
        return [self.stoi.get(tok, self.default_index) for tok in tokens]

    def __call__(self, tokens):
        return self.lookup_indices(tokens)


class Data_Factory:
    def __init__(self, path: str, num_worker=4) -> None:
        self.path = path
        self.num_worker = num_worker

        self.ag_news_vocab = None
        self.ag_news_pad_idx = 0
        self.ag_news_unk_idx = 1
        
        self.listops_vocab = None
        self.listops_pad_idx = 0
        self.listops_unk_idx = 1
        
    def _build_vocab_from_dataset(self, dataset, min_freq=1, specials=("<pad>", "<unk>")):
        counter = Counter()
        for item in dataset:
            counter.update(item["tokens"])

        itos = list(specials)
        for token, freq in counter.items():
            if freq >= min_freq and token not in specials:
                itos.append(token)

        stoi = {tok: idx for idx, tok in enumerate(itos)}
        vocab = SimpleVocab(stoi=stoi, itos=itos, unk_token="<unk>")
        vocab.set_default_index(stoi["<unk>"])
        return vocab
    def build_ag_news_vocab(self, dataset, min_freq=1, special_tokens=("<pad>", "<unk>")):
        vocab = self._build_vocab_from_dataset(dataset, min_freq=min_freq, specials=special_tokens)
        self.ag_news_vocab = vocab
        self.ag_news_pad_idx = vocab["<pad>"]
        self.ag_news_unk_idx = vocab["<unk>"]
    
    def build_listops_vocab(self, dataset, min_freq=1, special_tokens=("<pad>", "<unk>")):
        vocab = self._build_vocab_from_dataset(dataset, min_freq=min_freq, specials=special_tokens)
        self.listops_vocab = vocab
        self.listops_pad_idx = vocab["<pad>"]
        self.listops_unk_idx = vocab["<unk>"]
        
    def ag_news_collate_fn(self, batch):
        if self.ag_news_vocab is None:
            raise ValueError("AG_NEWS vocab has not been built.")

        sequence_list = []
        lengths = []
        labels = []
        label_names = []
        texts = []

        for item in batch:
            token_ids = torch.tensor(self.ag_news_vocab(item["tokens"]), dtype=torch.long)
            sequence_list.append(token_ids)
            lengths.append(len(token_ids))
            labels.append(item["label_id"])
            label_names.append(item["label"])
            texts.append(item["text"])

        padded_sequences = pad_sequence(
            sequence_list,
            batch_first=True,
            padding_value=self.ag_news_pad_idx,
        )

        return {
            "input_ids": padded_sequences,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "text": texts,
        }
    
    def listops_collate_fn(self, batch):
        if self.listops_vocab is None:
            raise ValueError("Long ListOps vocab has not been built.")

        sequence_list = []
        lengths = []
        labels = []
        label_names = []
        texts = []

        for item in batch:
            token_ids = torch.tensor(self.listops_vocab(item["tokens"]), dtype=torch.long)
            sequence_list.append(token_ids)
            lengths.append(len(token_ids))
            labels.append(item["label_id"])
            label_names.append(item["label"])
            texts.append(item["text"])

        padded_sequences = pad_sequence(
            sequence_list,
            batch_first=True,
            padding_value=self.listops_pad_idx,
        )

        return {
            "input_ids": padded_sequences,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "text": texts,
        }

    def repeat_to_length(self, seq: torch.Tensor, target_length: int) -> torch.Tensor:
        if seq.dim() != 1:
            raise ValueError("seq must be a 1D tensor")

        seq_len = seq.size(0)
        if seq_len == 0:
            raise ValueError("Empty sequence is not allowed")

        if seq_len >= target_length:
            return seq[:target_length]

        repeat_times = (target_length + seq_len - 1) // seq_len
        repeated = seq.repeat(repeat_times)
        return repeated[:target_length]

    def text_repeat_collate_fn(self, batch, vocab, fixed_length: int):
        if vocab is None:
            raise ValueError("vocab has not been built")

        sequence_list = []
        orig_lengths = []
        labels = []
        texts = []

        for item in batch:
            token_ids = torch.tensor(vocab(item["tokens"]), dtype=torch.long)
            orig_lengths.append(token_ids.size(0))
            fixed_ids = self.repeat_to_length(token_ids, fixed_length)
            sequence_list.append(fixed_ids)
            labels.append(item["label_id"])
            texts.append(item["text"])

        input_ids = torch.stack(sequence_list, dim=0)

        return {
            "input_ids": input_ids,
            "lengths": torch.full((len(batch),), fixed_length, dtype=torch.long),
            "orig_lengths": torch.tensor(orig_lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "text": texts,
        }

    def ag_news_repeat_collate_fn(self, batch, fixed_length: int):
        return self.text_repeat_collate_fn(batch=batch, vocab=self.ag_news_vocab, fixed_length=fixed_length)
    
    def get_data_loader(
        self,
        data_name: str,
        mode: str,
        offline: bool = True,
        batch_size=128,
        path=None,
        need_vali=True,
        vali_ratio=0.1,
        split_seed=2025,
        seq_mode="pixel",
        download=True,   # HF datasets其实不需要这个参数，但保留接口兼容
        permute=True,
        permutation: Union[torch.Tensor, None] = None,
        max_length=None,
        min_freq=1,
        fixed_length=None,
        length_mode="pad",
        to_grayscale=True,
        normalize=False,
    ) -> torch.utils.data.DataLoader:
        
        """
        The default setting is to use "esc50 fold 1" as the test dataset, you can specify other folds.
    
        """

        data_name = data_name.lower()
        mode = mode.lower()

        if path is None:
            path = self.path

        if data_name == "sequential_mnist":
            if mode in {"train", "training"}:
                is_train = True
                shuffle = True
            elif mode in {"vali", "validation", "val"}:
                is_train = True
                shuffle = False
            elif mode in {"test", "testing"}:
                is_train = False
                shuffle = False
            else:
                raise ValueError("Invalid mode")

            dataset = data_wrapper.SequentialMNISTDataset(
                path=path,
                is_train=is_train,
                need_vali=need_vali,
                is_download=download,
                seq_mode=seq_mode,
                permute=permute,
                permutation=permutation
            )

            if mode in {"train", "training", "vali", "validation", "val"} and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator = torch.Generator().manual_seed(split_seed)
                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset, [train_size, vali_size], generator=generator
                )
                dataset = train_dataset if mode in {"train", "training"} else vali_dataset

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_worker,
                pin_memory=torch.cuda.is_available(),
            )
            return loader
        
        elif data_name == "cifar10":
            if mode in {"train", "training"}:
                is_train = True
                shuffle = True
            elif mode in {"vali", "validation", "val"}:
                is_train = True
                shuffle = False
            elif mode in {"test", "testing"}:
                is_train = False
                shuffle = False
            else:
                raise ValueError("Invalid mode")

            dataset = data_wrapper.SequentialCIFAR10Dataset(
                path=path,
                is_train=is_train,
                need_vali=need_vali,
                is_download=download,
                seq_mode=seq_mode,
                permute=permute,
                permutation=permutation,
                to_grayscale=to_grayscale,   # LRA-style default
                normalize=normalize,
            )

            if mode in {"train", "training", "vali", "validation", "val"} and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator = torch.Generator().manual_seed(split_seed)
                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset, [train_size, vali_size], generator=generator
                )
                dataset = train_dataset if mode in {"train", "training"} else vali_dataset

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_worker,
                pin_memory=torch.cuda.is_available(),
            )
            return loader

        elif data_name == "ag_news":
            if length_mode == "pad":
                collate_fn = self.ag_news_collate_fn
            elif length_mode == "repeat":
                if fixed_length is None:
                    raise ValueError("fixed_length must be provided when length_mode='repeat'")
                collate_fn = lambda batch: self.ag_news_repeat_collate_fn(batch, fixed_length=fixed_length)
            else:
                raise ValueError("length_mode must be 'pad' or 'repeat'")

            if mode in {"train", "training"}:
                split = "train"
                shuffle = True
            elif mode in {"vali", "validation", "val"}:
                split = "train"
                shuffle = False
            elif mode in {"test", "testing"}:
                split = "test"
                shuffle = False
            else:
                raise ValueError("Invalid mode")

            dataset = data_wrapper.AGNewsDataset(
                path=path,
                split=split,
                max_length=max_length,
                offline_first=offline,
            )

            if self.ag_news_vocab is None:
                vocab_dataset = data_wrapper.AGNewsDataset(
                    path=path,
                    split="train",
                    max_length=max_length,
                    offline_first=offline,
                )
                self.build_ag_news_vocab(vocab_dataset, min_freq=min_freq)

            if mode in {"train", "training", "vali", "validation", "val"} and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator = torch.Generator().manual_seed(split_seed)
                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset, [train_size, vali_size], generator=generator
                )
                dataset = train_dataset if mode in {"train", "training"} else vali_dataset

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_worker,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_fn,
            )
            return loader

        elif data_name == "long_listops":
            if length_mode == "pad":
                collate_fn = self.listops_collate_fn
            else:
                raise ValueError("Long ListOps currently only supports length_mode='pad'")

            if mode in {"train", "training"}:
                split = "train"
                shuffle = True
            elif mode in {"vali", "validation", "val"}:
                split = "val"
                shuffle = False
            elif mode in {"test", "testing"}:
                split = "test"
                shuffle = False
            else:
                raise ValueError("Invalid mode")

            dataset = data_wrapper.LongListOpsDataset(
                path=path,
                split=split,
                max_length=max_length,
                offline_first=offline,
                # If you have other tasks in the future, you can change them to parameters
                task_name="basic",   
            )

            if self.listops_vocab is None:
                vocab_dataset = data_wrapper.LongListOpsDataset(
                    path=path,
                    split="train",
                    max_length=max_length,
                    offline_first=offline,
                    task_name="basic",
                )
                self.build_listops_vocab(vocab_dataset, min_freq=min_freq)

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=(mode in {"train", "training"}),
                num_workers=self.num_worker,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_fn,
            )
            return loader
        
        else:
            raise ValueError("Invalid data name")