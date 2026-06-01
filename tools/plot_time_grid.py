import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

SAVE_DIR = "./figures/efficiency"
SAVE_NAME = "efficiency_2x3_nature_bars_clean"
DPI = 600

# Runtime error bars:
# False -> use the reported standard deviations directly
# True  -> convert SD to SEM by dividing by sqrt(N_SEEDS)
USE_SEM = False
N_SEEDS = 5

# ============================================================
# Data
# ============================================================

lengths = [800, 900, 1000, 1100, 1200]

methods = [
    "BPTT",
    "SPTT-Hybrid",
    "SPTT-Endpoint",
    "TBPTT-4",
    "SPTT-Window-4",
]

# Nature-like muted palette
colors = {
    "BPTT": "#7BA6D0",          # muted blue
    "SPTT-Hybrid": "#6FBE73",   # muted green
    "SPTT-Endpoint": "#B07AA1", # muted purple-pink
    "TBPTT-4": "#E6A35A",       # muted orange
    "SPTT-Window-4": "#9A9A9A", # muted gray
}

# -------------------------
# 1) Runtime per framework gradient call (ms)
# -------------------------
per_call_time_mean = np.array([
    [30.7509, 35.5895, 41.4101, 44.9965, 49.1006],   # BPTT
    [10.2239, 11.8319, 13.0953, 13.9999, 15.4547],   # SPTT-Hybrid
    [0.9559,  0.9625,  0.9279,  0.9524,  0.9460],    # SPTT-Endpoint
    [8.1500,  9.1859, 10.2736, 11.2120, 12.2651],    # TBPTT-4
    [3.1988,  3.4629,  3.8835,  4.0685,  5.0016],    # SPTT-Window-4
])

per_call_time_std = np.array([
    [0.0892, 1.5909, 0.5998, 0.0896, 0.3889],
    [0.0571, 0.1843, 0.0033, 0.0284, 0.0147],
    [0.0069, 0.0051, 0.0168, 0.0122, 0.0125],
    [0.0146, 0.0106, 0.0196, 0.0197, 0.0158],
    [0.0045, 0.0037, 0.0044, 0.0048, 0.0060],
])

# -------------------------
# 2) FLOPs per framework gradient call (GFLOPs)
# deterministic -> no error bars
# -------------------------
per_call_flops = np.array([
    [644.2450944, 724.7757, 805.306, 885.837, 966.368],
    [35.8980,     40.3719,  44.856,  49.339,  53.823],
    [0.0645,       0.0645,   0.0645,  0.0645,  0.0645],
    [161.061,     181.194,  201.326, 221.459, 241.591],
    [8.987,        10.108,   11.228,  12.349,  13.470],
])

# -------------------------
# 3) Cumulative framework runtime to final performance (s)
# -------------------------
cum_time_mean = np.array([
    [230.5635, 245.3332, 269.4987, 379.2004, 560.3989],
    [59.0086,  67.2204,  74.4274,  77.7679,  96.2982],
    [5.6209,   5.6289,   5.5489,   5.6289,   5.6064],
    [167.8475, 187.9152, 271.4062, 228.7856, 250.4262],
    [63.8207,  69.9889,  76.7392,  83.6086, 102.0357],
])

cum_time_std = np.array([
    [14.2376, 18.9643, 9.0700, 1.2225, 26.0200],
    [1.4264,  2.4687,  3.8754, 1.7766,  1.8360],
    [0.0332,  0.0113,  0.1131, 0.0218,  0.0431],
    [4.5983,  2.9483,  7.6456, 3.1316,  3.6399],
    [1.1606,  0.0156,  2.4597, 1.9677,  1.3637],
])

# -------------------------
# 4) Cumulative framework FLOPs to final performance (TFLOPs)
# deterministic -> no error bars
# -------------------------
cum_flops = np.array([
    [4339.634, 4576.959, 5085.510, 7458.750, 11391.500],
    [211.525,   220.956,  264.373,  270.033,   339.890],
    [0.380,       0.380,    0.380,    0.380,     0.380],
    [3254.730, 3661.570, 5424.540, 4475.250,  4882.090],
    [181.606,   204.257,  226.908,  249.559,   272.210],
])

# ============================================================
# Helper functions
# ============================================================

def setup_nature_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def maybe_to_sem(err):
    if USE_SEM:
        return err / np.sqrt(N_SEEDS)
    return err


def format_axis(ax, ylabel=None, log_scale=False):
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels([str(x) for x in lengths])

    if log_scale:
        ax.set_yscale("log")

    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)

    ax.tick_params(axis="both", direction="out", width=0.75, length=3.0)

    ax.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.16)
    ax.grid(axis="x", visible=False)


def grouped_bar_panel(
    ax,
    values,
    errors,
    ylabel,
    panel_letter,
    log_scale=False,
):
    n_methods, n_groups = values.shape
    x = np.arange(n_groups)

    width = 0.15
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    for i, method in enumerate(methods):
        y = values[i]
        yerr = None if errors is None else maybe_to_sem(errors[i])

        ax.bar(
            x + offsets[i],
            y,
            width=width * 0.92,
            color=colors[method],
            edgecolor="#4A4A4A",
            linewidth=0.32,
            label=method,
            yerr=yerr,
            capsize=3.2 if yerr is not None else 0.0,
            error_kw=dict(
                ecolor="#202020",
                elinewidth=1.1,
                capthick=1.1,
            ),
            zorder=3,
        )

    format_axis(ax, ylabel=ylabel, log_scale=log_scale)

    ax.set_title(
        f"{panel_letter}",
        loc="left",
        pad=10,
        fontweight="bold",
        fontsize=16,
    )


# ============================================================
# Main figure
# ============================================================

def main():
    setup_nature_style()
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12.6, 6.9),
        facecolor="white",
    )

    # Column headers
    fig.text(
        0.22, 0.93,
        "Runtime",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.52, 0.93,
        "FLOPs",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.82, 0.93,
        "FLOPs (log scale)",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
    )

    # Row headers
    fig.text(
        0.025, 0.69,
        "Per-call cost",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.0,
        fontweight="bold",
    )
    fig.text(
        0.025, 0.37,
        "Cumulative cost",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.0,
        fontweight="bold",
    )

    # -------------------------
    # Row 1: per-call cost
    # -------------------------
    grouped_bar_panel(
        ax=axes[0, 0],
        values=per_call_time_mean,
        errors=per_call_time_std,
        ylabel="Runtime (ms)",
        panel_letter="a",
        log_scale=False,
    )

    grouped_bar_panel(
        ax=axes[0, 1],
        values=per_call_flops,
        errors=None,
        ylabel="FLOPs (GFLOPs)",
        panel_letter="b",
        log_scale=False,
    )

    grouped_bar_panel(
        ax=axes[0, 2],
        values=per_call_flops,
        errors=None,
        ylabel="FLOPs (GFLOPs, log)",
        panel_letter="c",
        log_scale=True,
    )

    # -------------------------
    # Row 2: cumulative cost
    # -------------------------
    grouped_bar_panel(
        ax=axes[1, 0],
        values=cum_time_mean,
        errors=cum_time_std,
        ylabel="Runtime (s)",
        panel_letter="d",
        log_scale=False,
    )

    grouped_bar_panel(
        ax=axes[1, 1],
        values=cum_flops,
        errors=None,
        ylabel="FLOPs (TFLOPs)",
        panel_letter="e",
        log_scale=False,
    )

    grouped_bar_panel(
        ax=axes[1, 2],
        values=cum_flops,
        errors=None,
        ylabel="FLOPs (TFLOPs, log)",
        panel_letter="f",
        log_scale=True,
    )

    # X labels
    for ax in axes[1, :]:
        ax.set_xlabel("Original sequence length")

    for ax in axes[0, :]:
        ax.set_xlabel("")

    # Figure-level legend outside the whole panel area
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], edgecolor="#4A4A4A", linewidth=0.32)
        for m in methods
    ]

    fig.legend(
        handles,
        methods,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.86,
        bottom=0.11,
        wspace=0.28,
        hspace=0.32,
    )

    # Save
    base = os.path.join(SAVE_DIR, SAVE_NAME)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.jpg", dpi=DPI, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Saved:")
    print(f"  {base}.pdf")
    print(f"  {base}.svg")
    print(f"  {base}.png")
    print(f"  {base}.jpg")


if __name__ == "__main__":
    main()