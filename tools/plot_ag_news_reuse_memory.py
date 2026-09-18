"""Plot AG News efficiency using embedded experimental measurements.

Run directly: python tools/plot_ag_news_reuse_memory.py
Dependencies: numpy, matplotlib. No external data files are required.
Error bars preserve the reported errors without conversion to SEM.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Configuration
SAVE_DIR = Path(__file__).resolve().parents[1] / 'figures' / 'efficiency'
DPI = 600
LENGTHS = [800, 900, 1000, 1100, 1200]
METHODS = ['BPTT', 'SPTT', 'SPTT-End', 'SPTT-Streaming']
DISPLAY_NAMES = ['BPTT', 'SPTT', 'SPTT-Endpoint', 'SPTT-Reuse-Memory']
COLORS = ['#7BA6D0', '#6FBE73', '#B07AA1', '#E6A35A']

# Embedded measurements from the experiment summary.
# Rows follow METHODS; columns follow LENGTHS. Errors use the same units as means.
# Units: batch runtime in seconds, total runtime in minutes, GPU memory in GiB.

BATCH_TIME_MEAN = np.array([
    [0.6001396, 0.6759802, 0.7419334000000001, 0.8202143, 0.8902589000000001],  # BPTT
    [0.6750475, 0.7570996, 0.8378599000000001, 0.9207998000000001, 1.0045795000000002],  # SPTT
    [0.5112645, 0.5719257, 0.6285498, 0.6920704, 0.7480939],  # SPTT-End
    [0.6949598, 0.7790834, 0.8586354, 0.943625, 1.0322331],  # SPTT-Streaming
], dtype=float)

BATCH_TIME_ERROR = np.array([
    [0.0005635, 0.0020244, 0.0011084, 4.38e-05, 0.0010719000000000002],  # BPTT
    [0.0007012, 0.0046779000000000005, 0.0007796, 0.0022053, 0.0025659000000000003],  # SPTT
    [0.0009593000000000001, 0.00024370000000000001, 0.0005393, 0.0021159, 0.0008999],  # SPTT-End
    [0.0008451, 0.0022427000000000002, 0.0024496, 0.0017258, 0.0014757000000000001],  # SPTT-Streaming
], dtype=float)

TOTAL_TIME_MEAN = np.array([
    [63.17028552833333, 87.77297019666666, 80.69969432333333, 97.83768431833333, 99.946403505],  # BPTT
    [61.575586236666666, 71.70465820833333, 76.42678905000001, 83.99228738333333, 91.63439091166666],  # SPTT
    [48.42714687833334, 56.18217075166667, 61.74454138, 67.98438624166667, 70.86542491],  # SPTT-End
    [63.39191723666667, 71.06539021333333, 78.32185823166667, 86.074326535, 94.15686663833334],  # SPTT-Streaming
], dtype=float)

TOTAL_TIME_ERROR = np.array([
    [11.969755981666667, 17.032270091666668, 11.163933445, 8.13381646, 0.12033657],  # BPTT
    [0.06395948166666666, 3.313256668333333, 0.07111089, 0.20115962666666667, 0.23404981833333335],  # SPTT
    [2.4457879666666664, 0.023937015, 0.05298011833333333, 0.20784803833333332, 3.7969348299999996],  # SPTT-End
    [0.07708582500000001, 0.20456904, 0.223445015, 0.15741866666666668, 0.13460715166666667],  # SPTT-Streaming
], dtype=float)

PEAK_MEMORY_MEAN = np.array([
    [3.7823783203125, 4.197712109375, 4.61314501953125, 5.0288328125, 5.444537109375],  # BPTT
    [4.98504716796875, 5.5514591796875, 6.11618876953125, 6.68416669921875, 7.24780341796875],  # SPTT
    [3.7878134765625, 4.2031880859375, 4.61863720703125, 5.03446650390625, 5.4506931640625],  # SPTT-End
    [3.88536142578125, 4.314066796875, 4.7428865234375, 5.17202275390625, 5.60120478515625],  # SPTT-Streaming
], dtype=float)

PEAK_MEMORY_ERROR = np.array([
    [8.59375e-06, 7.03125e-06, 5.95703125e-06, 3.3203125e-06, 0.0],  # BPTT
    [0.0, 2.63671875e-06, 0.0, 0.0, 0.0],  # SPTT
    [2.5390625e-06, 0.0, 0.0, 0.0, 2.5390625e-06],  # SPTT-End
    [0.0, 0.0, 0.0, 0.0, 0.0],  # SPTT-Streaming
], dtype=float)

PANELS = [
    ('Runtime per batch (s)', BATCH_TIME_MEAN, BATCH_TIME_ERROR),
    ('Total training runtime (min)', TOTAL_TIME_MEAN, TOTAL_TIME_ERROR),
    ('Peak GPU memory (GiB)', PEAK_MEMORY_MEAN, PEAK_MEMORY_ERROR),
]


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


def main():
    lengths, panels = LENGTHS, PANELS
    setup_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.45), facecolor='white')
    width = 0.15
    x = np.arange(len(lengths))
    for index, (ax, (label, means, errors)) in enumerate(zip(axes, panels)):
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, means[i],
                   width=width * 0.92, color=color, edgecolor='#4A4A4A', linewidth=0.32,
                   label=DISPLAY_NAMES[i], yerr=errors[i], capsize=3.2,
                   error_kw=dict(ecolor='#202020', elinewidth=1.1, capthick=1.1), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in lengths])
        ax.set_xlabel('Sequence length')
        ax.set_ylabel(label)
        ax.set_ylim(0, float(np.max(means + errors)) * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
        ax.tick_params(axis='both', direction='out', width=0.75, length=3)
        ax.grid(axis='y', linestyle='-', linewidth=0.35, alpha=0.16)
        ax.grid(axis='x', visible=False)
        ax.set_title(chr(ord('a') + index), loc='left', pad=10, fontweight='bold', fontsize=16)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='#4A4A4A', linewidth=0.32)
               for c in COLORS]
    fig.legend(handles, DISPLAY_NAMES, loc='upper center', bbox_to_anchor=(0.53, 0.99),
               ncol=4, frameon=False, handlelength=1.2, columnspacing=1.0,
               handletextpad=0.45, borderaxespad=0)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.17, wspace=0.30)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    base = SAVE_DIR / 'ag_news_reuse_memory_1x3'
    for suffix in ('jpg',):
        path = base.with_suffix('.' + suffix)
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white',
                    pil_kwargs={'quality': 95, 'subsampling': 0})
        print(path)
    # Small preview for visual review; the JPG remains at 600 dpi.
    fig.savefig(base.with_name(base.name + '_preview.png'), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Validated {len(panels) * len(METHODS) * len(lengths)} bars; SPTT-Streaming displayed as SPTT-Reuse-Memory.')


if __name__ == '__main__':
    main()
