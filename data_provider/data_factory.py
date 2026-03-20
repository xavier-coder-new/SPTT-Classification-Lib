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
from torchtext.vocab import build_vocab_from_iterator

class Data_Factory:
    def __init__(self, path: str, num_worker=4, sample_rate=16000, n_mels=64) -> None:
        self.path = path
        self.num_worker = num_worker
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            # The audio signal is converted to 128-dimensional Mel spectral characteristics at each time point.
            n_mels=n_mels,
        )
        self.count = 0
        
        self.imdb_vocab = None
        self.imdb_pad_idx = 0
        self.imdb_unk_idx = 1
        
        self.ag_news_vocab = None
        self.ag_news_pad_idx = 0
        self.ag_news_unk_idx = 1
        
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
        
    def build_imdb_vocab(self, dataset, min_freq=1, special_tokens=["<pad>", "<unk>"]):
        """
        Build vocabulary for IMDB dataset using torchtext
        
        min_frq: minimum frequency for a token to be included in the vocabulary
        special_tokens: list of special tokens to be added to the vocabulary
        
        min_freq = 1 means that all tokens that appear at least once in the dataset will be included in the vocabulary.
        min_freq = 5 means that only tokens that appear at least five times in the dataset will be included in the vocabulary.
        
        Args:
            dataset: IMDB dataset
            min_freq: minimum frequency for a token to be included in the vocabulary
            special_tokens: list of special tokens to be added to the vocabulary
            
        """
        def yield_tokens():
            for item in dataset:
                yield item["tokens"]
        
        vocab = build_vocab_from_iterator(
            yield_tokens(),
            min_freq=min_freq,
            special_tokens=special_tokens,
            special_first=True,  # Ensure special tokens are at the beginning of the vocabulary
        )
        
        vocab.set_default_index(vocab["<unk>"])
        self.imdb_vocab = vocab        
        self.imdb_pad_idx = vocab["<pad>"]
        self.imdb_unk_idx = vocab["<unk>"]
    
    def imdb_collate_fn(self, batch):
        """
        Prepare batch for IMDB text classification.
        
        Input batch:
            [
                {"text": str, "tokens": list[str], "label": str, "label_id": int},
                ...
            ]

        Returns:
            {
                "input_ids": [B, L],
                "lengths": [B],
                "label_id": [B],
                "label": list[str],
                "text": list[str],
            }
            
            {
                "input_ids": tensor([
                    [25, 134, 67, 892],      # Sample 1
                    [456, 789, 0, 0],        # Sample 2（pad to same length）
                    [12, 345, 678, 0]        # Sample 3（pad to same length）
                ]),                          # Shape: [B, L_max]
                
                "lengths": tensor([4, 2, 3]),           # Each sentence's true length
                "label_id": tensor([1, 0, 1]),          # Label ID
                "label": ["positive", "negative", "positive"],  # Label name
                "text": ["This movie is great", "Terrible film", "I love it"]  # Original text
            }
            
        """
        if self.imdb_vocab is None:
            raise ValueError("IMDB vocabulary has not been built. Please call build_imdb_vocab() first.")
        
        sequence_list = []
        lengths = []
        labels = []
        label_names = []
        texts = []
        
        for item in batch:
            """
            "item" is a sample in the batch, with the following structure:
                {
                    "text": str,           # original text
                    "tokens": list[str],   # list of tokenized words
                    "label": str,          # label name (e.g., "positive", "negative")
                    "label_id": int        # label ID (e.g., 0, 1)
                }
            """
            tokens = item["tokens"]
            
            # self.imdb_vocab(tokens) converts the list of words to the list of ids
            # For example: ["this", "movie", "is", "great"] → [25, 134, 67, 892]
            # Then convert to PyTorch tensor
            tokens_ids = torch.tensor(self.imdb_vocab(tokens), dtype=torch.long)
            
            sequence_list.append(tokens_ids)
            lengths.append(len(tokens_ids))
            labels.append(item["label_id"])
            label_names.append(item["label"])
            texts.append(item["text"])
            
        padded_sequences = pad_sequence(
            sequence_list,
            batch_first=True,
            padding_value=self.imdb_pad_idx,
        ) # [B, L]
        
        return {
            "input_ids": padded_sequences,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "text": texts,
        }

    def build_ag_news_vocab(self, dataset, min_freq=1, specials=["<pad>", "<unk>"]):
        def yield_tokens():
            for item in dataset:
                yield item["tokens"]

        vocab = build_vocab_from_iterator(
            yield_tokens(),
            min_freq=min_freq,
            specials=specials,
            special_first=True,
        )
        vocab.set_default_index(vocab["<unk>"])
        self.ag_news_vocab = vocab
        self.ag_news_pad_idx = vocab["<pad>"]
        self.ag_news_unk_idx = vocab["<unk>"]
    
    def ag_news_collate_fn(self, batch):
        """
        Normal collate fn for AG_NEWS
        Returns:
            {
                "input_ids": [B, L],
                "lengths": [B],
                "label_id": [B],
                "label": [B],   # original label values
                "text": list[str],
            }
        """
        if self.ag_news_vocab is None:
            raise ValueError("AG_NEWS vocab has not been built.")

        sequence_list = []
        lengths = []
        labels = []
        raw_labels = []
        texts = []

        for item in batch:
            token_ids = torch.tensor(self.ag_news_vocab(item["tokens"]), dtype=torch.long)
            sequence_list.append(token_ids)
            lengths.append(len(token_ids))
            labels.append(item["label_id"])
            raw_labels.append(item["label"])
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
            "label": torch.tensor(raw_labels, dtype=torch.long),
            "text": texts,
        }

    def repeat_to_length(self, seq: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Repeat a 1D tensor and truncate to target_length.

        Example:
            seq = [1,2,3], target_length=8
            -> [1,2,3,1,2,3,1,2]
        """
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
        """
        Ablation collate fn:
        Do NOT use padding token.
        Instead, repeat each sample itself until fixed_length, then truncate.

        Returns:
            {
                "input_ids": [B, fixed_length],
                "lengths": [B],           # 全部等于 fixed_length
                "orig_lengths": [B],      # 原始长度
                "label_id": [B],
                "text": list[str],
            }
        """
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

        input_ids = torch.stack(sequence_list, dim=0)  # [B, fixed_length]

        return {
            "input_ids": input_ids,
            "lengths": torch.full((len(batch),), fixed_length, dtype=torch.long),
            "orig_lengths": torch.tensor(orig_lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "text": texts,
        }
        
    def imdb_repeat_collate_fn(self, batch, fixed_length: int):
        return self.text_repeat_collate_fn(
            batch=batch,
            vocab=self.imdb_vocab,
            fixed_length=fixed_length,
        )
        
    def ag_news_repeat_collate_fn(self, batch, fixed_length: int):
        return self.text_repeat_collate_fn(
            batch=batch,
            vocab=self.ag_news_vocab,
            fixed_length=fixed_length,
        )
    
    def get_data_loader(self, data_name: str, mode: str, batch_size=128, path=None, 
                        need_vali=True, vali_ratio=0.1, split_seed=2025, seq_mode="pixel", 
                        download=True, permute=True, permutation: Union[torch.Tensor, None] = None,
                        max_length=None, min_freq=1, fixed_length=None, length_mode="pad") -> torch.utils.data.DataLoader:
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
            elif mode in {"vali", "validation", "val"}:
                is_train = True
                shuffle = False
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
            
            if mode in {"train", "training", "vali", "validation", "val"} and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator = torch.Generator().manual_seed(split_seed)

                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset,
                    [train_size, vali_size],
                    generator=generator,
                )
                if mode in {"train", "training"}:
                    dataset = train_dataset
                    ic(f"Loading training dataset (count: {self.count})")
                else:
                    ic(f"Loading validation dataset (count: {self.count})")
                    dataset = vali_dataset

            ic(f"Loading test dataset (count: {self.count})")
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
        
        elif data_name == "imdb":
            if length_mode == "pad":
                collate_fn = self.imdb_collate_fn
            elif length_mode == "repeat":
                ic("Using repeat mode for IMDB")
                if fixed_length is None:
                    raise ValueError("fixed_length must be provided when length_mode='repeat'")
                collate_fn = lambda batch: self.imdb_repeat_collate_fn(batch, fixed_length=fixed_length)
            else:
                raise ValueError("length_mode must be 'pad' or 'repeat'")
            
            if mode in {"train", "training"}:
                split = "train"
                shuffle = True
            elif mode in {"test", "testing"}:
                split = "test"
                shuffle = False
            elif mode in {"val", "validation", "vali"}:
                split = "train"
                shuffle = False
            else:
                raise ValueError("Invalid mode")
            
            dataset = data_wrapper.IMDBDataset(
                path=path,
                split=split,
                max_length=max_length,
            )
            
            # vocab must be built based on the training set
            if self.imdb_vocab is None:
                vocab_dataset = data_wrapper.IMDBDataset(
                    path=path,
                    split="train",
                    max_length=max_length,
                )
                
                self.build_imdb_vocab(vocab_dataset, min_freq)
                
            # divide the datasets into train and validation
            if split == "train" and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator  = torch.Generator().manual_seed(split_seed)
                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset, 
                    [train_size, vali_size],
                    generator=generator,
                )

                if mode in {"train", "training"}:
                    loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_worker,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=collate_fn,
                    )
                    return loader
                    
                elif mode in {"vali", "validation", "val"}:
                    loader = DataLoader(
                        dataset=vali_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=True,
                        num_workers=self.num_worker,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=collate_fn,
                    )
                    return loader
            else:
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
        
        elif data_name == "ag_news":
            if length_mode == "pad":
                collate_fn = self.ag_news_collate_fn
            elif length_mode == "repeat":
                ic("Using repeat mode for AG News")
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
            )

            if self.ag_news_vocab is None:
                vocab_dataset = data_wrapper.AGNewsDataset(
                    path=path,
                    split="train",
                    max_length=max_length,
                )
                self.build_ag_news_vocab(vocab_dataset, min_freq=min_freq)

            if mode in {"train", "training", "vali", "validation", "val"} and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator = torch.Generator().manual_seed(split_seed)

                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset,
                    [train_size, vali_size],
                    generator=generator,
                )

                if mode in {"train", "training"}:
                    dataset = train_dataset
                else:
                    dataset = vali_dataset

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
            
        else:
            raise ValueError("Invalid data name")
    
