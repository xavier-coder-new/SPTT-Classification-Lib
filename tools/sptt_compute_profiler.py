# tools/sptt_compute_profiler.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

from scipy import stats
import torch
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
import math


@dataclass
class OnlineStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    total: float = 0.0

    def update(self, x: float):
        x = float(x)
        self.n += 1
        self.total += x

        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def sample_std(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass
class SPTTComputeRecord:
    epoch: int
    batch_idx: int
    chunk_idx: int
    layer_idx: int

    model_name: str
    data_name: str
    mode_name: str
    architecture: str

    sequence_length: int
    chunk_length: int
    batch_size: int

    input_dim: int
    hidden_dim: int
    gate_multiplier: int
    rank_k: int

    block_len_total: int
    slide_window_nums: int
    num_internal_blocks: int

    local_sptt_time_ms: float
    local_sptt_flops: float


class SPTTComputeProfiler:
    def __init__(
        self, 
        enabled: bool = True, 
        keep_raw: bool = False,
        profile_epoch: int | None = 1,
    ):
        self.enabled = enabled
        self.keep_raw = keep_raw
        # None means all epochs, otherwise only profile the specified epoch
        self.profile_flops_epoch = profile_epoch
        
        self.records: List[SPTTComputeRecord] = []
        
        self._current_complete_key = None
        self._current_complete_time_ms = 0.0
        self._current_complete_flops = 0.0
        self._current_complete_layer_calls = 0
        self._current_rank_k = None
        self._current_complete_flops_profiled = False

        self.epoch_stats = {}

        self.epoch = 0
        self.batch_idx = 0
        self.chunk_idx = 0
        self.num_epochs_for_estimate = 1

        self.model_name = ""
        self.data_name = ""
        self.mode_name = ""
        self.architecture = ""

        self.sequence_length = 0
        self.chunk_length = 0
        self.batch_size = 0
        
    def set_num_epochs_for_estimate(self, num_epochs: int):
        self.num_epochs_for_estimate = int(num_epochs)

    def set_context(
        self,
        *,
        epoch: int,
        batch_idx: int,
        chunk_idx: int,
        model_name: str,
        data_name: str,
        sequence_length: int,
        chunk_length: int,
        batch_size: int,
        mode_name: str | None = None,
        architecture: str | None = None,
    ):
        self.epoch = int(epoch)
        self.batch_idx = int(batch_idx)
        self.chunk_idx = int(chunk_idx)

        self.model_name = str(model_name)
        self.data_name = str(data_name)

        self.sequence_length = int(sequence_length)
        self.chunk_length = int(chunk_length)
        self.batch_size = int(batch_size)

        if mode_name is None:
            if "End" in self.model_name or "Endpoint" in self.model_name:
                mode_name = "Endpoint"
            elif "Window" in self.model_name:
                mode_name = "Window"
            else:
                mode_name = "Hybrid"

        if architecture is None:
            if "GRU" in self.model_name:
                architecture = "GRU"
            elif "LSTM" in self.model_name:
                architecture = "LSTM"
            else:
                architecture = "Unknown"

        self.mode_name = mode_name
        self.architecture = architecture

    @staticmethod
    def sync_if_cuda(device):
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def matmul_flops(m: int, n: int, p: int) -> float:
        """
        [m, n] @ [n, p] -> [m, p]
        multiply-add counted as 2 FLOPs.
        """
        return 2.0 * m * n * p

    @staticmethod
    def qr_flops(m: int, k: int) -> float:
        """
        Reduced QR for m x k, m >= k:
        Householder QR approx: 2*m*k^2 - 2/3*k^3.
        """
        return max(0.0, 2.0 * m * k * k - (2.0 / 3.0) * k ** 3)

    @staticmethod
    def inverse_flops(k: int) -> float:
        return (2.0 / 3.0) * k ** 3

    @classmethod
    def estimate_sptt_cell_flops(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        gate_multiplier: int,
        rank_k: int,
        total_T: int,
        slide_window_nums: int,
    ) -> Dict[str, float]:

        I = int(input_dim)
        H = int(hidden_dim)
        G = int(gate_multiplier) * H
        K = int(rank_k)
        T = int(total_T)
        S = int(slide_window_nums)

        if T <= 0:
            return {
                "single_layer_sptt_flops": 0.0,
                "num_internal_blocks": 0,
                "internal_block_len": 0,
            }

        t = max(1, T // S)
        num_internal_blocks = max(1, T // t)

        total = 0.0

        total += cls.inverse_flops(K) * 2

        for i in range(1, num_internal_blocks + 1):
            start_idx = (i - 1) * t
            end_idx = min(i * t, T)
            B = end_idx - start_idx

            if B <= 0:
                continue

            # scale_factor_right_ih = delta_ih_block @ Delta_matrix_ih
            total += cls.matmul_flops(B, G, K)

            # scale_factor_left_ih = activation_input @ X_matrix_ih
            total += cls.matmul_flops(B, I, K)

            # scale_factor_right_hh = delta_hh_block @ Delta_matrix_hh
            total += cls.matmul_flops(B, G, K)

            # scale_factor_left_hh = activation_hx @ X_matrix_hh
            total += cls.matmul_flops(B, H, K)

            # X_ih_update = activation_input.T @ scale_factor_right_ih @ inv_Sigma_ih
            total += cls.matmul_flops(I, B, K)
            total += cls.matmul_flops(I, K, K)

            # Delta_ih_update = delta_ih_block.T @ scale_factor_left_ih @ inv_Sigma_ih
            total += cls.matmul_flops(G, B, K)
            total += cls.matmul_flops(G, K, K)

            # X_hh_update = activation_hx.T @ scale_factor_right_hh @ inv_Sigma_hh
            total += cls.matmul_flops(H, B, K)
            total += cls.matmul_flops(H, K, K)

            # Delta_hh_update = delta_hh_block.T @ scale_factor_left_hh @ inv_Sigma_hh
            total += cls.matmul_flops(G, B, K)
            total += cls.matmul_flops(G, K, K)

            # QR: X_ih, Delta_ih, X_hh, Delta_hh
            total += cls.qr_flops(I, K)
            total += cls.qr_flops(G, K)
            total += cls.qr_flops(H, K)
            total += cls.qr_flops(G, K)

            # sign alignment / element-wise scaling，low-order terms
            total += 2.0 * G * K
            total += 2.0 * I * K
            total += 2.0 * H * K

            # Sigma_ih_product
            total += cls.matmul_flops(B, G, K)
            total += cls.matmul_flops(B, I, K)
            total += 3.0 * B * K

            # Sigma_hh_product
            total += cls.matmul_flops(B, G, K)
            total += cls.matmul_flops(B, H, K)
            total += 3.0 * B * K

            # inverse after Sigma update
            total += cls.inverse_flops(K) * 2

        # grad_w_ih = (X_ih @ Sigma_ih @ Delta_ih.T).T
        total += cls.matmul_flops(I, K, K)
        total += cls.matmul_flops(I, K, G)

        # grad_w_hh = (X_hh @ Sigma_hh @ Delta_hh.T).T
        total += cls.matmul_flops(H, K, K)
        total += cls.matmul_flops(H, K, G)

        return {
            "single_layer_sptt_flops": float(total),
            "num_internal_blocks": int(num_internal_blocks),
            "internal_block_len": int(t),
        }
    
    def add_local_record(
        self,
        *,
        layer_idx: int,
        input_dim: int,
        hidden_dim: int,
        gate_multiplier: int,
        rank_k: int,
        block_len_total: int,
        slide_window_nums: int,
        elapsed_ms: float,
    ):
        if not self.enabled:
            return

        elapsed_ms = float(elapsed_ms)

        do_profile_flops = (
            self.profile_flops_epoch is None
            or int(self.epoch) == int(self.profile_flops_epoch)
        )

        local_flops = 0.0
        est = {
            "single_layer_sptt_flops": 0.0,
            "num_internal_blocks": 0,
        }

        if do_profile_flops:
            est = self.estimate_sptt_cell_flops(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                gate_multiplier=gate_multiplier,
                rank_k=rank_k,
                total_T=block_len_total,
                slide_window_nums=slide_window_nums,
            )
            local_flops = float(est["single_layer_sptt_flops"])

        complete_key = self._get_complete_key()

        if self._current_complete_key is None:
            self._current_complete_key = complete_key

        elif complete_key != self._current_complete_key:
            self._flush_current_complete_gradient(rank_k=self._current_rank_k)
            self._current_complete_key = complete_key

        self._current_complete_time_ms += elapsed_ms

        if do_profile_flops:
            self._current_complete_flops += local_flops
            self._current_complete_flops_profiled = True

        self._current_complete_layer_calls += 1
        self._current_rank_k = int(rank_k)

        if self.keep_raw:
            self.records.append(
                SPTTComputeRecord(
                    epoch=self.epoch,
                    batch_idx=self.batch_idx,
                    chunk_idx=self.chunk_idx,
                    layer_idx=int(layer_idx),
                    model_name=self.model_name,
                    data_name=self.data_name,
                    mode_name=self.mode_name,
                    architecture=self.architecture,
                    sequence_length=self.sequence_length,
                    chunk_length=self.chunk_length,
                    batch_size=self.batch_size,
                    input_dim=int(input_dim),
                    hidden_dim=int(hidden_dim),
                    gate_multiplier=int(gate_multiplier),
                    rank_k=int(rank_k),
                    block_len_total=int(block_len_total),
                    slide_window_nums=int(slide_window_nums),
                    num_internal_blocks=int(est["num_internal_blocks"]),
                    local_sptt_time_ms=elapsed_ms,
                    local_sptt_flops=local_flops,
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_csv(self, path):
        if not self.keep_raw:
            print("Raw record saving is disabled because keep_raw=False.")
            return path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    def complete_gradient_summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df

        group_cols = [
            "model_name",
            "data_name",
            "mode_name",
            "architecture",
            "epoch",
            "batch_idx",
            "chunk_idx",
            "sequence_length",
            "chunk_length",
            "batch_size",
            "rank_k",
        ]

        per_backward = (
            df.groupby(group_cols, as_index=False)
            .agg(
                single_grad_time_ms=("local_sptt_time_ms", "sum"),
                single_grad_flops=("local_sptt_flops", "sum"),
                num_layer_sptt_calls=("local_sptt_flops", "count"),
            )
        )

        return per_backward
    
    def epoch_summary(self) -> pd.DataFrame:
        self._flush_current_complete_gradient(rank_k=self._current_rank_k)

        rows = []

        for stats in self.epoch_stats.values():
            time_stats = stats["time_stats"]
            flops_stats = stats["flops_stats"]

            has_profiled_flops = stats["num_profiled_complete_grad_calls"] > 0

            rows.append({
                "model_name": stats["model_name"],
                "data_name": stats["data_name"],
                "mode_name": stats["mode_name"],
                "architecture": stats["architecture"],
                "epoch": stats["epoch"],
                "sequence_length": stats["sequence_length"],
                "chunk_length": stats["chunk_length"],
                "batch_size": stats["batch_size"],
                "rank_k": stats["rank_k"],

                "single_grad_time_ms_mean": time_stats.mean,
                "single_grad_time_ms_std": time_stats.sample_std,
                "epoch_framework_time_ms": stats["epoch_framework_time_ms"],
                "epoch_framework_time_sec": stats["epoch_framework_time_sec"],

                "is_flops_profiled_epoch": has_profiled_flops,
                "single_grad_flops_mean": (
                    flops_stats.mean if has_profiled_flops else pd.NA
                ),
                "single_grad_flops_std": (
                    flops_stats.sample_std if has_profiled_flops else pd.NA
                ),
                "profiled_epoch_framework_flops": (
                    stats["profiled_epoch_framework_flops"] if has_profiled_flops else pd.NA
                ),

                "num_complete_grad_calls": stats["num_complete_grad_calls"],
                "num_layer_sptt_calls": stats["num_layer_sptt_calls"],
                "num_profiled_complete_grad_calls": stats["num_profiled_complete_grad_calls"],
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        total_time_ms = df["epoch_framework_time_ms"].sum()
        df["total_framework_time_ms_measured"] = total_time_ms
        df["total_framework_time_sec_measured"] = total_time_ms / 1000.0

        profiled_rows = df[df["is_flops_profiled_epoch"] == True]

        if len(profiled_rows) > 0:
            profiled_epoch_flops = profiled_rows["profiled_epoch_framework_flops"].iloc[0]
            df["profiled_epoch_for_flops"] = int(profiled_rows["epoch"].iloc[0])
        else:
            df["profiled_epoch_for_flops"] = pd.NA
            df["total_framework_flops_est"] = pd.NA

        return df

    def save_complete_gradient_summary(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.complete_gradient_summary()
        df.to_csv(path, index=False)
        return path

    def save_epoch_summary(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.epoch_summary()
        df.to_csv(path, index=False)
        return path
    
    
    def _get_complete_key(self):
        return (
            self.model_name,
            self.data_name,
            self.mode_name,
            self.architecture,
            self.epoch,
            self.batch_idx,
            self.chunk_idx,
            self.sequence_length,
            self.chunk_length,
            self.batch_size,
        )


    def _get_epoch_key(self):
        return (
            self.model_name,
            self.data_name,
            self.mode_name,
            self.architecture,
            self.epoch,
            self.sequence_length,
            self.chunk_length,
            self.batch_size,
        )


    def _ensure_epoch_stats(self, epoch_key, rank_k):
        if epoch_key not in self.epoch_stats:
            (
                model_name,
                data_name,
                mode_name,
                architecture,
                epoch,
                sequence_length,
                chunk_length,
                batch_size,
            ) = epoch_key

            self.epoch_stats[epoch_key] = {
                "model_name": model_name,
                "data_name": data_name,
                "mode_name": mode_name,
                "architecture": architecture,
                "epoch": epoch,
                "sequence_length": sequence_length,
                "chunk_length": chunk_length,
                "batch_size": batch_size,
                "rank_k": int(rank_k),

                "time_stats": OnlineStats(),
                "flops_stats": OnlineStats(),

                "epoch_framework_time_ms": 0.0,
                "epoch_framework_time_sec": 0.0,
                "profiled_epoch_framework_flops": 0.0,
                "num_complete_grad_calls": 0, 
                "num_layer_sptt_calls": 0, 
                
                "num_profiled_complete_grad_calls": 0, 
            }

        return self.epoch_stats[epoch_key]

    def _flush_current_complete_gradient(self, rank_k=None):
        if self._current_complete_key is None:
            return

        if self._current_complete_layer_calls <= 0:
            self._current_complete_key = None
            self._current_complete_time_ms = 0.0
            self._current_complete_flops = 0.0
            self._current_complete_layer_calls = 0
            return

        (
            model_name,
            data_name,
            mode_name,
            architecture,
            epoch,
            batch_idx,
            chunk_idx,
            sequence_length,
            chunk_length,
            batch_size,
        ) = self._current_complete_key

        epoch_key = (
            model_name,
            data_name,
            mode_name,
            architecture,
            epoch,
            sequence_length,
            chunk_length,
            batch_size,
        )

        stats = self._ensure_epoch_stats(epoch_key, rank_k if rank_k is not None else -1)

        stats["time_stats"].update(self._current_complete_time_ms)
        stats["epoch_framework_time_ms"] += self._current_complete_time_ms
        stats["epoch_framework_time_sec"] += self._current_complete_time_ms / 1000.0

        stats["num_complete_grad_calls"] += 1
        stats["num_layer_sptt_calls"] += self._current_complete_layer_calls

        if self._current_complete_flops_profiled:
            stats["flops_stats"].update(self._current_complete_flops)
            stats["profiled_epoch_framework_flops"] += self._current_complete_flops
            stats["num_profiled_complete_grad_calls"] += 1

        self._current_complete_key = None
        self._current_complete_time_ms = 0.0
        self._current_complete_flops = 0.0
        self._current_complete_layer_calls = 0
        self._current_complete_flops_profiled = False
        self._current_rank_k = None


def plot_sptt_compute_metrics(summary_csv_or_df, save_dir, *, title="SPTT compute profile"):
    if isinstance(summary_csv_or_df, (str, Path)):
        df = pd.read_csv(summary_csv_or_df)
    else:
        df = summary_csv_or_df.copy()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ["model_name", "mode_name", "architecture"]

    def build_label(row):
        parts = [str(row[c]) for c in label_cols if c in row and pd.notna(row[c])]
        return "-".join(parts)

    df["plot_label"] = df.apply(build_label, axis=1)

    metrics = [
        (
            "single_grad_time_ms_mean",
            "single_grad_time_ms_std",
            "Single complete-gradient time (ms)",
            "single_grad_time_ms",
        ),
        (
            "single_grad_flops_mean",
            "single_grad_flops_std",
            "Single complete-gradient FLOPs",
            "single_grad_flops",
        ),
        (
            "epoch_framework_flops",
            None,
            "Epoch-level framework FLOPs",
            "epoch_framework_flops",
        ),
    ]

    for mean_col, std_col, ylabel, filename in metrics:
        plt.figure(figsize=(6.4, 4.2))

        for label, sub in df.groupby("plot_label"):
            sub = sub.sort_values("sequence_length")
            x = sub["sequence_length"].to_numpy()
            y = sub[mean_col].to_numpy()

            plt.plot(x, y, marker="o", linewidth=1.8, label=label)

            if std_col is not None and std_col in sub.columns:
                err = sub[std_col].fillna(0.0).to_numpy()
                plt.fill_between(x, y - err, y + err, alpha=0.15)

        plt.xlabel("Sequence length")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {ylabel}")
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / f"{filename}.svg", format="svg")
        plt.close()