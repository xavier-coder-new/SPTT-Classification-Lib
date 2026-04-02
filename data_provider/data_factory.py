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
    def __init__(self, path: str, num_worker=4, 
                 sample_rate=16000, n_mels=64,
                 n_fft=1024, hop_length=512) -> None:
        print(f"n_mels: {n_mels}, sample_rate: {sample_rate}, n_fft: {n_fft}, hop_length: {hop_length}")
        self.path = path
        self.num_worker = num_worker
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        self.imdb_vocab = None
        self.imdb_pad_idx = 0
        self.imdb_unk_idx = 1

        self.ag_news_vocab = None
        self.ag_news_pad_idx = 0
        self.ag_news_unk_idx = 1
        
        self.byte_imdb_pad_idx = 0
        self.byte_imdb_vocab_size = 257
        self.byte_imdb_eos_idx = None


    # def speech_commands_collate_fn(self, batch):
    #     feature_list = []
    #     lengths = []
    #     labels = []
    #     sample_rates = []
    #     label_names = []
    #     speaker_ids = []
    #     utterance_numbers = []

    #     for item in batch:
    #         waveform = item["waveform"]  # [1, T]
    #         mel = self.mel_transform(waveform)          # [1, n_mels, time_frames]
    #         mel = mel.squeeze(0).transpose(0, 1)       # [time_frames, n_mels]

    #         feature_list.append(mel)
    #         lengths.append(mel.size(0))
    #         labels.append(item["label_id"])
    #         sample_rates.append(item["sample_rate"])
    #         label_names.append(item["label"])
    #         speaker_ids.append(item["speaker_id"])
    #         utterance_numbers.append(item["utterance_number"])

    #     padded_features = pad_sequence(feature_list, batch_first=True)

    #     return {
    #         "features": padded_features,
    #         "lengths": torch.tensor(lengths, dtype=torch.long),
    #         "label_id": torch.tensor(labels, dtype=torch.long),
    #         "label": label_names,
    #         "sample_rate": torch.tensor(sample_rates, dtype=torch.long),
    #         "speaker_id": speaker_ids,
    #         "utterance_number": torch.tensor(utterance_numbers, dtype=torch.long),
    #     }

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
        
    def esc50_collate_fn(self, batch):
        feature_list = []
        lengths = []
        labels = []
        sample_rates = []
        label_names = []
        filenames = []
        folds = []

        for item in batch:
            waveform = item["waveform"]  # [1, T]
            mel = self.mel_transform(waveform)          # [1, n_mels, time_frames]
            mel = mel.squeeze(0).transpose(0, 1)       # [time_frames, n_mels]

            feature_list.append(mel)
            lengths.append(mel.size(0))
            labels.append(item["label_id"])
            sample_rates.append(item["sample_rate"])
            label_names.append(item["label"])
            filenames.append(item["filename"])
            folds.append(item["fold"])

        padded_features = pad_sequence(feature_list, batch_first=True)

        return {
            "features": padded_features,                         # [B, T, n_mels]
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "sample_rate": torch.tensor(sample_rates, dtype=torch.long),
            "filename": filenames,
            "fold": torch.tensor(folds, dtype=torch.long),
        }

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

    def build_imdb_vocab(self, dataset, min_freq=1, special_tokens=("<pad>", "<unk>")):
        vocab = self._build_vocab_from_dataset(dataset, min_freq=min_freq, specials=special_tokens)
        self.imdb_vocab = vocab
        self.imdb_pad_idx = vocab["<pad>"]
        self.imdb_unk_idx = vocab["<unk>"]

    def build_ag_news_vocab(self, dataset, min_freq=1, special_tokens=("<pad>", "<unk>")):
        vocab = self._build_vocab_from_dataset(dataset, min_freq=min_freq, specials=special_tokens)
        self.ag_news_vocab = vocab
        self.ag_news_pad_idx = vocab["<pad>"]
        self.ag_news_unk_idx = vocab["<unk>"]

    def imdb_collate_fn(self, batch):
        if self.imdb_vocab is None:
            raise ValueError("IMDB vocabulary has not been built.")

        sequence_list = []
        lengths = []
        labels = []
        label_names = []
        texts = []

        for item in batch:
            token_ids = torch.tensor(self.imdb_vocab(item["tokens"]), dtype=torch.long)
            sequence_list.append(token_ids)
            lengths.append(len(token_ids))
            labels.append(item["label_id"])
            label_names.append(item["label"])
            texts.append(item["text"])

        padded_sequences = pad_sequence(
            sequence_list,
            batch_first=True,
            padding_value=self.imdb_pad_idx,
        )

        return {
            "input_ids": padded_sequences,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "text": texts,
        }

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

    def imdb_repeat_collate_fn(self, batch, fixed_length: int):
        return self.text_repeat_collate_fn(batch=batch, vocab=self.imdb_vocab, fixed_length=fixed_length)

    def ag_news_repeat_collate_fn(self, batch, fixed_length: int):
        return self.text_repeat_collate_fn(batch=batch, vocab=self.ag_news_vocab, fixed_length=fixed_length)

    def imdb_byte_collate_fn(self, batch, fixed_length=None, pad_value=0):
        sequence_list = []
        lengths = []
        labels = []
        label_names = []
        texts = []

        for item in batch:
            byte_ids = torch.tensor(item["byte_ids"], dtype=torch.long)

            if fixed_length is not None:
                if byte_ids.size(0) >= fixed_length:
                    byte_ids = byte_ids[:fixed_length]
                    seq_len = fixed_length
                else:
                    seq_len = byte_ids.size(0)
            else:
                seq_len = byte_ids.size(0)

            sequence_list.append(byte_ids)
            lengths.append(seq_len)
            labels.append(item["label_id"])
            label_names.append(item["label"])
            texts.append(item["text"])

        if fixed_length is not None:
            padded_sequences = []
            for seq in sequence_list:
                if seq.size(0) < fixed_length:
                    pad_len = fixed_length - seq.size(0)
                    seq = torch.cat(
                        [seq, torch.full((pad_len,), pad_value, dtype=torch.long)],
                        dim=0,
                    )
                else:
                    seq = seq[:fixed_length]
                padded_sequences.append(seq)

            input_ids = torch.stack(padded_sequences, dim=0)
            out_lengths = torch.tensor(lengths, dtype=torch.long)
        else:
            input_ids = pad_sequence(
                sequence_list,
                batch_first=True,
                padding_value=pad_value,
            )
            out_lengths = torch.tensor(lengths, dtype=torch.long)

        return {
            "input_ids": input_ids,                 # [B, T]
            "lengths": out_lengths,                 # [B]
            "label_id": torch.tensor(labels, dtype=torch.long),
            "label": label_names,
            "text": texts,
        }
    
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
        esc50_fold=1,
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
                to_grayscale=True,   # LRA-style default
                normalize=False,
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

        elif data_name == "google_speech":
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
                # sample_rate=self.sample_rate,
                # config_name="v0.02",
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
        
        elif data_name == "esc50":
            if mode in {"train", "training"}:
                shuffle = True
            elif mode in {"val", "validation", "vali"}:
                shuffle = False
            elif mode in {"test", "testing"}:
                shuffle = False
            else:
                raise ValueError("Invalid mode")

            dataset = data_wrapper.ESC50Dataset(
                path=path,
                mode=mode,
                sample_rate=self.sample_rate,
                fold=esc50_fold,
                offline_first=offline,
            )

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=(mode in {"train", "training"}),
                num_workers=self.num_worker,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.esc50_collate_fn,
            )
            return loader

        elif data_name == "imdb":
            if length_mode == "pad":
                collate_fn = self.imdb_collate_fn
            elif length_mode == "repeat":
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
                offline_first=offline,
            )

            if self.imdb_vocab is None:
                vocab_dataset = data_wrapper.IMDBDataset(
                    path=path,
                    split="train",
                    max_length=max_length,
                    offline_first=offline,
                )
                self.build_imdb_vocab(vocab_dataset, min_freq=min_freq)

            if split == "train" and need_vali:
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
        
        elif data_name == "byte_imdb":
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

            dataset = data_wrapper.ByteIMDBDataset(
                path=path,
                split=split,
                max_length=max_length,
                offline_first=offline,
                add_eos=False,
            )

            self.byte_imdb_pad_idx = dataset.pad_idx
            self.byte_imdb_vocab_size = dataset.byte_vocab_size
            self.byte_imdb_eos_idx = dataset.eos_idx

            if split == "train" and need_vali:
                train_size = int((1 - vali_ratio) * len(dataset))
                vali_size = len(dataset) - train_size
                generator = torch.Generator().manual_seed(split_seed)
                train_dataset, vali_dataset = torch.utils.data.random_split(
                    dataset, [train_size, vali_size], generator=generator
                )
                dataset = train_dataset if mode in {"train", "training"} else vali_dataset

            collate_fn = lambda batch: self.imdb_byte_collate_fn(
                batch=batch,
                fixed_length=fixed_length if fixed_length is not None else max_length,
                pad_value=self.byte_imdb_pad_idx,
            )

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