# tools/bptt_compute_profiler.py

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
import pandas as pd


@dataclass
class OnlineStats:
    """
    Welford online statistics.
    sample_std 使用样本标准差，即 ddof=1。
    """
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
class BPTTComputeRecord:
    """
    可选 raw record。
    默认 keep_raw=False 时不会保存，避免数据量过大。
    """
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

    统计口径：
        local record:
            每个 time step、每层的一次局部权重梯度计算。

        complete gradient:
            同一个 epoch / batch / chunk 内，
            所有 time step、所有层的局部权重梯度计算聚合为一次完整 BPTT 梯度计算。

        epoch summary:
            在线累计一个 epoch 内所有 complete gradient 的统计量。

    LSTM:
        gate_multiplier = 4

    GRU:
        gate_multiplier = 3
    """

    def __init__(self, 
            enabled: bool = True, 
            keep_raw: bool = False,
            profile_epoch: int | None = 1,
        ):
        self.enabled = enabled
        self.keep_raw = keep_raw
        # None means estimate FLOPs for all epochs;
        # otherwise estimate FLOPs only for the specified epoch.
        self.profile_flops_epoch = profile_epoch

        # 默认不保存 raw，避免 BPTT 每个 time step 都记录导致内存爆炸
        self.records: List[BPTTComputeRecord] = []

        # 当前 complete-gradient 累计缓存
        self._current_complete_key = None
        self._current_complete_time_ms = 0.0
        self._current_complete_flops = 0.0
        self._current_complete_local_calls = 0
        self._current_complete_flops_profiled = False

        # epoch 级在线统计
        self.epoch_stats = {}

        # 当前上下文
        self.epoch = 0
        self.batch_idx = 0
        self.chunk_idx = 0
        self.num_epochs_for_estimate = 1

        self.model_name = ""
        self.data_name = ""
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
        architecture: str | None = None,
    ):
        """
        在每个 batch/chunk 的 forward 前调用。
        """
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
        """
        delta:  [B, G]
        inputs: [B, I]
        hx:     [B, H]
        G = gate_multiplier * H

        grad_w_ih = delta.T @ inputs = [G, B] @ [B, I]
        grad_w_hh = delta.T @ hx     = [G, B] @ [B, H]
        """
        B = int(batch_size)
        I = int(input_dim)
        H = int(hidden_dim)
        G = int(gate_multiplier) * H

        flops_ih = cls.matmul_flops(G, B, I)
        flops_hh = cls.matmul_flops(G, B, H)

        return float(flops_ih + flops_hh)

    def _get_complete_key(self):
        """
        一个 batch/chunk 对应一次完整 BPTT 梯度计算。
        """
        return (
            self.model_name,
            self.data_name,
            self.architecture,
            self.epoch,
            self.batch_idx,
            self.chunk_idx,
            self.sequence_length,
            self.chunk_length,
            self.batch_size,
        )

    def _get_epoch_key_from_complete_key(self, complete_key):
        (
            model_name,
            data_name,
            architecture,
            epoch,
            batch_idx,
            chunk_idx,
            sequence_length,
            chunk_length,
            batch_size,
        ) = complete_key

        return (
            model_name,
            data_name,
            architecture,
            epoch,
            sequence_length,
            chunk_length,
            batch_size,
        )

    def _ensure_epoch_stats(self, epoch_key):
        if epoch_key not in self.epoch_stats:
            (
                model_name,
                data_name,
                architecture,
                epoch,
                sequence_length,
                chunk_length,
                batch_size,
            ) = epoch_key

            self.epoch_stats[epoch_key] = {
                "model_name": model_name,
                "data_name": data_name,
                "architecture": architecture,
                "epoch": int(epoch),
                "sequence_length": int(sequence_length),
                "chunk_length": int(chunk_length),
                "batch_size": int(batch_size),

                "time_stats": OnlineStats(),
                "flops_stats": OnlineStats(),

                "epoch_framework_time_ms": 0.0,
                "epoch_framework_time_sec": 0.0,
                
                # FLOPs: estimated only for profiled epoch
                "profiled_epoch_framework_flops": 0.0,

                "num_complete_grad_calls": 0,
                "num_local_grad_calls": 0,

                "num_profiled_complete_grad_calls": 0,
                "num_profiled_local_grad_calls": 0,
            }

        return self.epoch_stats[epoch_key]

    def _flush_current_complete_gradient(self):
        """
        将当前 batch/chunk 内累计的所有 local BPTT 梯度计算
        聚合为一次 complete gradient，并更新 epoch 级统计。

        时间：所有 epoch 都统计。
        FLOPs：只在 profile_flops_epoch 指定的 epoch 统计。
        """
        if self._current_complete_key is None:
            return

        if self._current_complete_local_calls <= 0:
            self._current_complete_key = None
            self._current_complete_time_ms = 0.0
            self._current_complete_flops = 0.0
            self._current_complete_local_calls = 0
            self._current_complete_flops_profiled = False
            return

        epoch_key = self._get_epoch_key_from_complete_key(self._current_complete_key)
        stats = self._ensure_epoch_stats(epoch_key)

        # 1. 时间：所有 epoch 都统计
        stats["time_stats"].update(self._current_complete_time_ms)
        stats["epoch_framework_time_ms"] += self._current_complete_time_ms
        stats["epoch_framework_time_sec"] += self._current_complete_time_ms / 1000.0

        stats["num_complete_grad_calls"] += 1
        stats["num_local_grad_calls"] += self._current_complete_local_calls

        # 2. FLOPs：只在指定 profiled epoch 统计
        if self._current_complete_flops_profiled:
            stats["flops_stats"].update(self._current_complete_flops)
            stats["profiled_epoch_framework_flops"] += self._current_complete_flops
            stats["num_profiled_complete_grad_calls"] += 1
            stats["num_profiled_local_grad_calls"] += self._current_complete_local_calls

        self._current_complete_key = None
        self._current_complete_time_ms = 0.0
        self._current_complete_flops = 0.0
        self._current_complete_local_calls = 0
        self._current_complete_flops_profiled = False

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
        """
        每个 LSTM/GRU cell backward 中计算完 grad_w_ih / grad_w_hh 后调用一次。

        时间统计：
            所有 epoch 都记录。

        FLOPs 统计：
            只在 profile_flops_epoch 指定的 epoch 中估算；
            非目标 epoch 不重复计算 FLOPs。
        """
        if not self.enabled:
            return

        elapsed_ms = float(elapsed_ms)

        do_profile_flops = (
            self.profile_flops_epoch is None
            or int(self.epoch) == int(self.profile_flops_epoch)
        )

        local_flops = 0.0

        if do_profile_flops:
            local_flops = self.estimate_local_bptt_flops(
                batch_size=batch_size,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                gate_multiplier=gate_multiplier,
            )
            local_flops = float(local_flops)

        complete_key = self._get_complete_key()

        # 如果进入新的 batch/chunk，先把上一个 complete gradient 刷入 epoch 统计
        if self._current_complete_key is None:
            self._current_complete_key = complete_key

        elif complete_key != self._current_complete_key:
            self._flush_current_complete_gradient()
            self._current_complete_key = complete_key

        # 时间：所有 epoch 都累计
        self._current_complete_time_ms += elapsed_ms

        # FLOPs：只在指定 epoch 累计
        if do_profile_flops:
            self._current_complete_flops += local_flops
            self._current_complete_flops_profiled = True

        self._current_complete_local_calls += 1

        # 可选 raw record，默认关闭
        if self.keep_raw:
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
                    local_grad_time_ms=elapsed_ms,
                    local_grad_flops=local_flops,
                )
            )

    def epoch_summary(self) -> pd.DataFrame:
        """
        只从在线统计生成 epoch summary，不依赖 raw records。

        时间：
            所有 epoch 正常统计。

        FLOPs：
            只在 profile_flops_epoch 指定的 epoch 统计；
            total_framework_flops_est = profiled_epoch_framework_flops * num_epochs_for_estimate。
        """
        self._flush_current_complete_gradient()

        rows = []

        for stats in self.epoch_stats.values():
            time_stats = stats["time_stats"]
            flops_stats = stats["flops_stats"]

            has_profiled_flops = stats["num_profiled_complete_grad_calls"] > 0

            rows.append({
                "model_name": stats["model_name"],
                "data_name": stats["data_name"],
                "architecture": stats["architecture"],
                "epoch": stats["epoch"],
                "sequence_length": stats["sequence_length"],
                "chunk_length": stats["chunk_length"],
                "batch_size": stats["batch_size"],

                # time: every epoch
                "single_grad_time_ms_mean": time_stats.mean,
                "single_grad_time_ms_std": time_stats.sample_std,
                "epoch_framework_time_ms": stats["epoch_framework_time_ms"],
                "epoch_framework_time_sec": stats["epoch_framework_time_sec"],

                # FLOPs: profiled epoch only
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
                "num_local_grad_calls": stats["num_local_grad_calls"],
                "num_profiled_complete_grad_calls": stats["num_profiled_complete_grad_calls"],
                "num_profiled_local_grad_calls": stats["num_profiled_local_grad_calls"],
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # 所有 epoch 实测框架时间总和
        total_time_ms = df["epoch_framework_time_ms"].sum()
        df["total_framework_time_ms_measured"] = total_time_ms
        df["total_framework_time_sec_measured"] = total_time_ms / 1000.0

        # 只用 profiled epoch 的 FLOPs 估算总 FLOPs
        profiled_rows = df[df["is_flops_profiled_epoch"] == True]

        if len(profiled_rows) > 0:
            profiled_epoch_flops = profiled_rows["profiled_epoch_framework_flops"].iloc[0]
            profiled_epoch_id = int(profiled_rows["epoch"].iloc[0])

            df["profiled_epoch_for_flops"] = profiled_epoch_id
            # df["num_epochs_for_estimate"] = self.num_epochs_for_estimate
            # df["total_framework_flops_est"] = (
            #     profiled_epoch_flops * self.num_epochs_for_estimate
            # )
        else:
            df["profiled_epoch_for_flops"] = pd.NA
            # df["num_epochs_for_estimate"] = self.num_epochs_for_estimate
            df["total_framework_flops_est"] = pd.NA

        return df

    def save_epoch_summary(self, path):
        """
        建议只调用这个函数。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.epoch_summary()
        df.to_csv(path, index=False)
        return path

    # -----------------------------
    # 以下函数仅在 keep_raw=True 时有意义
    # -----------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.keep_raw:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_csv(self, path):
        """
        默认不建议调用。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.keep_raw:
            print("Raw BPTT record saving is disabled because keep_raw=False.")
            pd.DataFrame().to_csv(path, index=False)
            return path

        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    def complete_gradient_summary(self) -> pd.DataFrame:
        """
        为了兼容旧接口保留，但 keep_raw=False 时不建议使用。
        """
        if not self.keep_raw:
            return pd.DataFrame()

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

    def save_complete_gradient_summary(self, path):
        """
        默认不建议调用。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.keep_raw:
            print("Complete-gradient saving is disabled because keep_raw=False.")
            pd.DataFrame().to_csv(path, index=False)
            return path

        df = self.complete_gradient_summary()
        df.to_csv(path, index=False)
        return path