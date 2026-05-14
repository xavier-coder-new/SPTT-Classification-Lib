# tools/bptt_compute_profiler.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
import pandas as pd


@dataclass
class BPTTComputeRecord:
    epoch: int
    batch_idx: int
    chunk_idx: int
    layer_idx: int

    model_name: str
    data_name: str
    architecture: str

    sequence_length: int
    chunk_length: int
    batch_size: int

    input_dim: int
    hidden_dim: int
    gate_multiplier: int

    local_grad_time_ms: float
    local_grad_flops: float


class BPTTComputeProfiler:
    """
    通用 BPTT 计算框架 profiler。

    统计范围：
        grad_w_ih = delta.T @ inputs
        grad_w_hh = delta.T @ hx

    raw record 粒度：
        每个 time step、每层的一次局部权重梯度计算。

    complete-gradient summary 粒度：
        同一个 epoch / batch / chunk 内，
        所有 time step、所有层的局部权重梯度计算聚合为一次完整 BPTT 梯度计算。

    LSTM:
        gate_multiplier = 4

    GRU:
        gate_multiplier = 3
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: List[BPTTComputeRecord] = []

        self.epoch = 0
        self.batch_idx = 0
        self.chunk_idx = 0

        self.model_name = ""
        self.data_name = ""
        self.architecture = ""

        self.sequence_length = 0
        self.chunk_length = 0
        self.batch_size = 0

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

        if architecture is None:
            if "GRU" in self.model_name:
                architecture = "GRU"
            elif "LSTM" in self.model_name:
                architecture = "LSTM"
            else:
                architecture = "Unknown"

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

    @classmethod
    def estimate_local_bptt_flops(
        cls,
        *,
        batch_size: int,
        input_dim: int,
        hidden_dim: int,
        gate_multiplier: int,
    ) -> float:
        B = int(batch_size)
        I = int(input_dim)
        H = int(hidden_dim)
        G = int(gate_multiplier) * H

        # grad_w_ih = [G, B] @ [B, I]
        flops_ih = cls.matmul_flops(G, B, I)

        # grad_w_hh = [G, B] @ [B, H]
        flops_hh = cls.matmul_flops(G, B, H)

        return float(flops_ih + flops_hh)

    def add_local_record(
        self,
        *,
        layer_idx: int,
        input_dim: int,
        hidden_dim: int,
        gate_multiplier: int,
        batch_size: int,
        elapsed_ms: float,
    ):
        if not self.enabled:
            return

        local_flops = self.estimate_local_bptt_flops(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gate_multiplier=gate_multiplier,
        )

        self.records.append(
            BPTTComputeRecord(
                epoch=self.epoch,
                batch_idx=self.batch_idx,
                chunk_idx=self.chunk_idx,
                layer_idx=int(layer_idx),
                model_name=self.model_name,
                data_name=self.data_name,
                architecture=self.architecture,
                sequence_length=self.sequence_length,
                chunk_length=self.chunk_length,
                batch_size=int(batch_size),
                input_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                gate_multiplier=int(gate_multiplier),
                local_grad_time_ms=float(elapsed_ms),
                local_grad_flops=float(local_flops),
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    def complete_gradient_summary(self) -> pd.DataFrame:
        """
        将同一个 batch/chunk 中所有 time step、所有层的局部权重梯度计算
        聚合为一次完整 BPTT 梯度计算。
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        group_cols = [
            "model_name",
            "data_name",
            "architecture",
            "epoch",
            "batch_idx",
            "chunk_idx",
            "sequence_length",
            "chunk_length",
            "batch_size",
        ]

        per_backward = (
            df.groupby(group_cols, as_index=False)
            .agg(
                single_grad_time_ms=("local_grad_time_ms", "sum"),
                single_grad_flops=("local_grad_flops", "sum"),
                num_local_grad_calls=("local_grad_flops", "count"),
            )
        )

        return per_backward

    def epoch_summary(self) -> pd.DataFrame:
        """
        输出论文可用统计：
            single_grad_time_ms_mean ± single_grad_time_ms_std
            single_grad_flops_mean ± single_grad_flops_std
            epoch_framework_flops

        std 使用样本标准差 ddof=1。
        """
        per_backward = self.complete_gradient_summary()
        if per_backward.empty:
            return per_backward

        group_cols = [
            "model_name",
            "data_name",
            "architecture",
            "epoch",
            "sequence_length",
            "chunk_length",
        ]

        summary = (
            per_backward.groupby(group_cols, as_index=False)
            .agg(
                single_grad_time_ms_mean=("single_grad_time_ms", "mean"),
                single_grad_time_ms_std=("single_grad_time_ms", lambda x: x.std(ddof=1)),
                single_grad_flops_mean=("single_grad_flops", "mean"),
                single_grad_flops_std=("single_grad_flops", lambda x: x.std(ddof=1)),
                epoch_framework_flops=("single_grad_flops", "sum"),
                num_complete_grad_calls=("single_grad_flops", "count"),
            )
        )

        summary["single_grad_time_ms_std"] = summary["single_grad_time_ms_std"].fillna(0.0)
        summary["single_grad_flops_std"] = summary["single_grad_flops_std"].fillna(0.0)

        return summary

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