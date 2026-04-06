import torch
from pathlib import Path
from exp.exp_basic import Exp_basic
from data_provider.data_factory import Data_Factory
import time
from tools.utils import _setup_logger, calculate_accuracy, visual_loss
from rich.console import Console
from tools.TrainProgressDisplay import TrainingProgress
import torch.nn as nn
from torch.optim.adam import Adam
import torch.optim.lr_scheduler as lr_scheduler
from tools.EarlyStop import Earlystopping
import math
import numpy as np
import pandas as pd
import os
from icecream import ic


class Exp_audio_classification_5fold(Exp_basic):
    """
    A 5-fold friendly audio experiment class.

    Design goal:
    - Reuse your existing Data_Factory(..., esc50_fold=...)
    - Keep the train / validate / test logic almost unchanged
    - Add:
        1) single-fold training/testing
        2) automatic 5-fold loop for ESC-50
        3) mean/std aggregation

    Expected args used here:
        args.data_name
        args.batch_size
        args.num_worker
        args.sample_rate
        args.n_mels
        args.n_fft
        args.hop_length
        args.esc50_fold
        args.nsynth_label_type
        args.nsynth_config
        args.model
        args.exp_type
        args.seed
        args.krank
        args.truncate_num
        args.slide_window_nums
        args.epochs
        args.patience
        args.lr_rate
        args.use_rich
        args.save

    Optional new args:
        args.run_five_fold   -> bool, whether to run all 5 folds for ESC-50
        args.dataset_path    -> custom dataset root path
    """

    def __init__(self, args, device):
        super().__init__(args)
        self.device = device
        self.chunk_len = None

    def _resolve_dataset_path(self, data_name, default_path="./datasets"):
        if hasattr(self.args, "dataset_path") and self.args.dataset_path:
            return self.args.dataset_path

        if data_name == "google_speech":
            return "/home/leo/breeze/NMI/code/uoro_pytorch/datasets"

        return default_path

    def _get_loader(self, data_name, path="./datasets", esc50_fold=None):
        path = self._resolve_dataset_path(data_name, path)
        if esc50_fold is None:
            esc50_fold = getattr(self.args, "esc50_fold", 1)

        data_loader = Data_Factory(
            path=Path(path),
            num_worker=self.args.num_worker,
            sample_rate=self.args.sample_rate,
            n_mels=self.args.n_mels,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
        )

        train_loader = data_loader.get_data_loader(
            data_name=data_name,
            mode="train",
            batch_size=self.args.batch_size,
            download=True,
            esc50_fold=esc50_fold,
            nsynth_label_type=self.args.nsynth_label_type,
            nsynth_config=self.args.nsynth_config,
        )
        vali_loader = data_loader.get_data_loader(
            data_name=data_name,
            mode="vali",
            batch_size=self.args.batch_size,
            download=True,
            esc50_fold=esc50_fold,
            nsynth_label_type=self.args.nsynth_label_type,
            nsynth_config=self.args.nsynth_config,
        )
        test_loader = data_loader.get_data_loader(
            data_name=data_name,
            mode="test",
            batch_size=self.args.batch_size,
            download=True,
            esc50_fold=esc50_fold,
            nsynth_label_type=self.args.nsynth_label_type,
            nsynth_config=self.args.nsynth_config,
        )

        return train_loader, vali_loader, test_loader

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _build_export_path(self) -> Path:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        fold_suffix = ""
        if getattr(self.args, "data_name", "").lower() == "esc50":
            fold_suffix = f"/fold_{getattr(self.args, 'esc50_fold', 1)}"

        loss_path = (
            Path("loss_data")
            / self.args.exp_type
            / self.args.model
            / self.args.data_name
            / f"seed_{self.args.seed}"
            / f"krank_{self.args.krank}_Trun_{self.args.truncate_num}_Slide_{self.args.slide_window_nums}"
            / f"Epoch-{self.args.epochs}_Patience-{self.args.patience}"
        )

        if fold_suffix:
            loss_path = Path(str(loss_path) + fold_suffix)

        train_loss_path = loss_path / (
            f"train_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}"
            f"_krank-{self.args.krank}_truncate-{self.args.truncate_num}"
            f"_slide_{self.args.slide_window_nums}_{timestamp}.csv"
        )
        train_loss_path.parent.mkdir(parents=True, exist_ok=True)

        vali_loss_path = loss_path / (
            f"vali_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}"
            f"_krank-{self.args.krank}_truncate-{self.args.truncate_num}"
            f"_slide_{self.args.slide_window_nums}_{timestamp}.csv"
        )
        vali_loss_path.parent.mkdir(parents=True, exist_ok=True)

        visual_train_loss = loss_path / (
            f"train_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}"
            f"_krank-{self.args.krank}_truncate-{self.args.truncate_num}"
            f"_slide_{self.args.slide_window_nums}_{timestamp}.svg"
        )
        visual_vali_loss = loss_path / (
            f"vali_loss=Epoch-{self.args.epochs}_Patience-{self.args.patience}"
            f"_krank-{self.args.krank}_truncate-{self.args.truncate_num}"
            f"_slide_{self.args.slide_window_nums}_{timestamp}.svg"
        )
        visual_train_loss.parent.mkdir(parents=True, exist_ok=True)
        visual_vali_loss.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_path = (
            Path("checkpoints")
            / self.args.exp_type
            / self.args.model
            / self.args.data_name
            / f"seed_{self.args.seed}"
            / f"krank_{self.args.krank}_Trun_{self.args.truncate_num}_Slide_{self.args.slide_window_nums}"
            / f"Epoch-{self.args.epochs}_Patience-{self.args.patience}"
        )
        if fold_suffix:
            checkpoint_path = Path(str(checkpoint_path) + fold_suffix)

        checkpoint_path = checkpoint_path / f"checkpoint_{timestamp}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        return train_loss_path, vali_loss_path, visual_train_loss, visual_vali_loss, checkpoint_path

    def _split_time_steps(self, batch_x):
        if batch_x.dim() == 3:
            inputs = [batch_x[:, t, :] for t in range(batch_x.size(1))]
        elif batch_x.dim() == 2:
            inputs = [batch_x[:, t] for t in range(batch_x.size(1))]
        else:
            raise ValueError(f"Unsupported batch_x shape: {batch_x.shape}")
        return inputs

    def train_one_fold(self, fold=None):
        if fold is not None:
            self.args.esc50_fold = int(fold)

        self.file_logger, self.console_logger = _setup_logger(
            self.args.save, self.args, use_rich=self.args.use_rich
        )
        self.file_logger.info(f"Starting training with args: {self.args}")

        train_loader, vali_loader, test_loader = self._get_loader(
            self.args.data_name, esc50_fold=getattr(self.args, "esc50_fold", 1)
        )
        self.model = self._build_model().to(self.device)

        ic(f"Input type: {self.args.input_type}")
        ic(f"Current fold: {getattr(self.args, 'esc50_fold', 'N/A')}")

        (
            train_loss_path,
            vali_loss_path,
            visual_train_loss,
            visual_vali_loss,
            checkpoint_path,
        ) = self._build_export_path()

        if self.args.use_rich:
            console = Console()
            training_progress = TrainingProgress(total_epochs=self.args.epochs, console=console)
        else:
            raise ValueError("Please set use_rich to True to use rich progress bar")

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.args.lr_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6,
            last_epoch=-1,
        )

        epoch_count = 0
        first_train = True
        is_nan = False

        early_stopping = Earlystopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.epochs):
            self.model.train()

            epoch_count += 1
            batch_count = 0
            iter_count = 0

            epoch_correct_num = 0
            epoch_total_num = 0

            epoch_loss = []

            with training_progress.create_progress_bar("Training") as progress:
                task_id = progress.add_task(f"Epoch {epoch}", total=len(train_loader))

                for i, batch in enumerate(train_loader):
                    batch_x = batch["features"].to(self.device)
                    sequence_length = batch_x.size(1)
                    actual_length = batch["lengths"].to(self.device)
                    labels = batch["label_id"].to(self.device)

                    batch_x, batch_y = batch_x.to(self.device), labels.to(self.device)
                    batch_size = batch_x.size(0)

                    self.model.model.reset_state(batch_size)
                    self.model.reset_logits()

                    assert self.model.model.hx_list is None or all(
                        torch.all(hx == 0) for hx in self.model.model.hx_list
                    ), "Hidden state is not reset to zeros at the beginning of each batch"

                    batch_count += 1
                    iter_count += 1

                    num_chunks = 0
                    sequence_loss = 0.0

                    if self.args.truncate_num == 1:
                        chunk_len = sequence_length
                    else:
                        if self.args.truncate_num > sequence_length:
                            raise ValueError(
                                f"truncate_num {self.args.truncate_num} is greater than sequence length {sequence_length}"
                            )
                        chunk_len = math.ceil(sequence_length / self.args.truncate_num)
                        chunk_len = max(1, chunk_len)

                    for start in range(0, sequence_length, chunk_len):
                        end = min(start + chunk_len, sequence_length)
                        chunk_x = batch_x[:, start:end, :]
                        chunk_T = chunk_x.size(1)

                        inputs = self._split_time_steps(chunk_x)

                        remaining_length = (actual_length - start).clamp(min=0)
                        chunk_actual_length = remaining_length.clamp(max=chunk_T)
                        end_in_chunked = (remaining_length > 0) & (remaining_length <= chunk_T)

                        if self.args.model in {"SpttLSTM", "SpttGRU", "SpttLSTM_End", "SpttGRU_End"}:
                            self.model.reset_sptt_state(chunk_actual_length)

                        if first_train:
                            print(f"batch_x shape: {batch_x.shape}, chunk_x shape: {chunk_x.shape}")
                            print(f"inputs length: {len(inputs)}")
                            print(f"chunk_actual_length[:8]: {chunk_actual_length[:8]}")
                            print(f"end_in_chunk[:8]: {end_in_chunked[:8]}")
                            first_train = False
                            self.chunk_len = chunk_len

                        optimizer.zero_grad()

                        output = self.model(
                            inputs=inputs,
                            actual_length=chunk_actual_length,
                            sequence_length=chunk_T,
                            ended_in_chunk=end_in_chunked,
                        )

                        loss = self.criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()

                        sequence_loss += loss.item()
                        num_chunks += 1
                        self.model.model.detach_state()

                    batch_loss = sequence_loss / max(1, num_chunks)
                    epoch_loss.append(batch_loss)

                    batch_correct, batch_total = calculate_accuracy(output, batch_y)
                    batch_acc = batch_correct / batch_total

                    epoch_correct_num += batch_correct
                    epoch_total_num += batch_total

                    if np.isnan(batch_loss):
                        print(f"Batch loss is NaN. Batch count: {batch_count}, Iter count: {iter_count}")
                        self.file_logger.warning(
                            f"Training stopped due to NaN loss at epoch {epoch_count}, batch {batch_count}."
                        )
                        is_nan = True
                        break

                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Epoch {epoch_count} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}",
                    )

                    if batch_count % 20 == 0 or batch_count == 1:
                        training_progress.print_batch_info(
                            epoch=epoch_count,
                            batch=batch_count,
                            loss=batch_loss,
                            acc=batch_acc,
                            total_batches=len(train_loader),
                        )

                    self.file_logger.info(
                        f"Epoch {epoch_count}, Batch {batch_count}, Loss: {batch_loss:.4f}, "
                        f"Chunk_num: {num_chunks}, Chunk_length: {chunk_len}, Accuracy: {batch_acc:.4f}"
                    )

                train_loss_data = pd.DataFrame({"train_loss": epoch_loss})

                if os.path.exists(train_loss_path):
                    train_loss_data.to_csv(train_loss_path, mode="a", header=False, index=False)
                else:
                    train_loss_data.to_csv(train_loss_path, mode="w", header=True, index=False)

                if is_nan:
                    ic("NaN loss, stopping training immediately")
                    break

                avg_loss = np.average(epoch_loss)
                epoch_acc = epoch_correct_num / epoch_total_num
                training_progress.print_epoch_summary(
                    epoch=epoch_count,
                    avg_loss=avg_loss,
                    avg_acc=epoch_acc,
                    lr=optimizer.param_groups[0]["lr"],
                )

                vali_av_loss, vali_acc = self.validate(vali_loader, vali_loss_path)

                self.file_logger.info(
                    f"-----Epoch {epoch_count}, Average Loss: {avg_loss:.4f}, "
                    f"Train Accuracy: {epoch_acc:.4f}, Validation Accuracy: {vali_acc:.4f} -------"
                )

                early_stopping(vali_av_loss, self.model, checkpoint_path)
                scheduler.step()

                if early_stopping.early_stop:
                    ic("Early stopping")

                    train_loss_pd = pd.read_csv(train_loss_path)
                    visual_loss(
                        train_loss_pd["train_loss"].tolist(),
                        label="Training Loss",
                        name=visual_train_loss,
                    )

                    vali_loss_pd = pd.read_csv(vali_loss_path)
                    visual_loss(
                        vali_loss_pd["vali_loss"].tolist(),
                        label="Validation Loss",
                        name=visual_vali_loss,
                    )

                    self.file_logger.info(
                        f"Early stopping at epoch {epoch_count}. Best validation loss: {early_stopping.val_loss_min:.4f}"
                    )
                    break

        return checkpoint_path, test_loader, num_chunks

    def validate(self, vali_loader, vali_loss_csv_file):
        self.model.eval()
        total_loss = []
        correct_num = 0
        sample_num = 0

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x = batch["features"].to(self.device)
                sequence_length = batch_x.size(1)
                actual_length = batch["lengths"].to(self.device)
                labels = batch["label_id"].to(self.device)

                batch_x, batch_y = batch_x.to(self.device), labels.to(self.device)
                batch_size = batch_x.size(0)

                self.model.reset_logits()
                self.model.model.reset_state(batch_size)

                inputs = self._split_time_steps(batch_x)

                output = self.model(
                    inputs=inputs,
                    actual_length=actual_length,
                    sequence_length=sequence_length,
                    ended_in_chunk=torch.ones(
                        batch_size, dtype=torch.bool, device=self.device
                    ),
                )

                loss = self.criterion(output, batch_y)
                total_loss.append(loss.item())

                batch_correct, batch_total = calculate_accuracy(output, batch_y)
                correct_num += batch_correct
                sample_num += batch_total

        vali_loss_data = pd.DataFrame({"vali_loss": total_loss})

        if os.path.exists(vali_loss_csv_file):
            vali_loss_data.to_csv(vali_loss_csv_file, mode="a", header=False, index=False)
        else:
            vali_loss_data.to_csv(vali_loss_csv_file, mode="w", header=True, index=False)

        vali_acc = correct_num / sample_num
        total_loss = np.average(total_loss)
        return total_loss, vali_acc

    def test(self, best_model_path, test_loader, num_chunks):
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        sample_num = 0
        correct_num = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x = batch["features"].to(self.device)
                sequence_length = batch_x.size(1)
                actual_length = batch["lengths"].to(self.device)
                labels = batch["label_id"].to(self.device)

                batch_x, batch_y = batch_x.to(self.device), labels.to(self.device)
                batch_size = batch_x.size(0)

                self.model.reset_logits()
                self.model.model.reset_state(batch_size)

                inputs = self._split_time_steps(batch_x)

                output = self.model(
                    inputs=inputs,
                    actual_length=actual_length,
                    sequence_length=sequence_length,
                    ended_in_chunk=torch.ones(
                        batch_size, dtype=torch.bool, device=self.device
                    ),
                )

                batch_correct, batch_total = calculate_accuracy(output, batch_y)
                sample_num += batch_total
                correct_num += batch_correct

        test_acc = correct_num / sample_num
        print("Test Accuracy: ", test_acc, self.chunk_len)
        self.file_logger.info(
            f"Test Accuracy: {test_acc:.4f} with chunk length {self.chunk_len} and chunk num {num_chunks}"
        )
        return test_acc

    def run_single_fold(self, fold=None):
        """
        Run one fold and return the test accuracy.
        """
        if fold is not None:
            self.args.esc50_fold = int(fold)

        checkpoint_path, test_loader, num_chunks = self.train_one_fold(
            fold=getattr(self.args, "esc50_fold", 1)
        )
        test_acc = self.test(checkpoint_path, test_loader, num_chunks)
        return test_acc

    def run_five_fold(self, folds=(1, 2, 3, 4, 5)):
        """
        Standard ESC-50 style 5-fold evaluation:
        - each fold is used once as test fold
        - validation fold is already handled inside your ESC50Dataset logic
        """
        if getattr(self.args, "data_name", "").lower() != "esc50":
            raise ValueError("run_five_fold is intended for ESC-50 only.")

        results = []
        summary_dir = Path("results") / self.args.exp_type / self.args.model / self.args.data_name
        summary_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        summary_path = summary_dir / f"esc50_5fold_summary_{timestamp}.csv"

        for fold in folds:
            print("=" * 80)
            print(f"Starting ESC-50 fold {fold}")
            print("=" * 80)

            self.args.esc50_fold = int(fold)
            fold_acc = self.run_single_fold(fold=fold)

            results.append({
                "fold": fold,
                "test_acc": fold_acc,
            })

        df = pd.DataFrame(results)
        mean_acc = float(df["test_acc"].mean())
        std_acc = float(df["test_acc"].std(ddof=0))

        df.loc[len(df)] = {
            "fold": "mean",
            "test_acc": mean_acc,
        }
        df.loc[len(df)] = {
            "fold": "std",
            "test_acc": std_acc,
        }

        df.to_csv(summary_path, index=False)

        print("=" * 80)
        print("ESC-50 5-fold finished")
        print(f"Mean ACC: {mean_acc:.6f}")
        print(f"Std  ACC: {std_acc:.6f}")
        print(f"Saved summary to: {summary_path}")
        print("=" * 80)

        return {
            "fold_results": results,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "summary_path": str(summary_path),
        }

    def run(self):
        """
        Unified entry:
        - for ESC-50 + run_five_fold=True -> run full 5-fold
        - otherwise -> run a normal single experiment
        """
        run_five_fold = getattr(self.args, "run_five_fold", False)
        if getattr(self.args, "data_name", "").lower() == "esc50" and run_five_fold:
            print(f">>>>>>>start five-fold training : {self.args.model} on {self.args.data_name} with seed {self.args.seed}<<<<<<")
            return self.run_five_fold()
        else:
            print(f">>>>>>>start single training : {self.args.model} on {self.args.data_name} with seed {self.args.seed}<<<<<<")
            test_acc = self.run_single_fold(fold=getattr(self.args, "esc50_fold", 1))
            return {"test_acc": test_acc}