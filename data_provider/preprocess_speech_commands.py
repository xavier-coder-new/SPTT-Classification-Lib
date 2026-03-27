import os
from pathlib import Path
import torch
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm


def preprocess_split(
    cache_dir,
    output_dir,
    split,
    config_name="v0.02",
    sample_rate=16000,
    n_mels=64,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading split={split} ...")
    ds = load_dataset(
        "google/speech_commands",
        config_name,
        split=split,
        cache_dir=str(cache_dir),
    )

    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
    )

    label_names = ds.features["label"].names
    index_to_label = {i: name for i, name in enumerate(label_names)}

    all_items = []

    for item in tqdm(ds, desc=f"processing {split}"):
        audio_info = item["audio"]

        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # [1, T]
        elif waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        mel = mel_transform(waveform)          # [1, n_mels, time]
        mel = mel.squeeze(0).transpose(0, 1)   # [time, n_mels]

        label_id = int(item["label"])
        label_name = index_to_label[label_id]

        speaker_id = str(item.get("speaker_id", "unknown"))
        try:
            utterance_number = int(item.get("utterance_id", 0))
        except (TypeError, ValueError):
            utterance_number = 0

        all_items.append({
            "feature": mel.cpu(),  # [time, n_mels]
            "length": int(mel.size(0)),
            "label_id": label_id,
            "label": label_name,
            "speaker_id": speaker_id,
            "utterance_number": utterance_number,
            "sample_rate": sample_rate,
        })

    save_path = output_dir / f"speech_commands_{split}_mel.pt"
    torch.save(all_items, save_path)
    print(f"Saved to: {save_path}")
    print(f"Num samples: {len(all_items)}")


if __name__ == "__main__":
    cache_dir = "/mnt/hard_disk/weihao/SPTT-Classification-Lib/datasets"
    output_dir = "/mnt/hard_disk/weihao/SPTT-Classification-Lib/datasets/speech_commands_mel"

    for split in ["train", "validation", "test"]:
        preprocess_split(
            cache_dir=cache_dir,
            output_dir=output_dir,
            split=split,
            config_name="v0.02",
            sample_rate=16000,
            n_mels=64,
        )