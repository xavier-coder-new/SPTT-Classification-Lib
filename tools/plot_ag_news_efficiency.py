"""Plot AG News efficiency using embedded experimental measurements.

Run directly: python tools/plot_ag_news_efficiency.py
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
METHODS = ['BPTT', 'SPTT', 'SPTT-End', 'TBPTT', 'SPTT-Window']
DISPLAY_NAMES = ['BPTT', 'SPTT', 'SPTT-Endpoint', 'TBPTT', 'SPTT-Window']
COLORS = ['#7BA6D0', '#6FBE73', '#B07AA1', '#E6A35A', '#9A9A9A']

# Embedded measurements from the experiment summary.
# Rows follow METHODS; columns follow LENGTHS. Errors use the same units as means.
# Units: batch runtime in seconds, total runtime in minutes, GPU memory in GiB.

BATCH_TIME_MEAN = np.array([
    [0.6001396, 0.6759802, 0.7419334000000001, 0.8202143, 0.8902589000000001],  # BPTT
    [0.6750475, 0.7570996, 0.8378599000000001, 0.9207998000000001, 1.0045795000000002],  # SPTT
    [0.5112645, 0.5719257, 0.6285498, 0.6920704, 0.7480939],  # SPTT-End
    [0.6526376, 0.7241757, 0.7956636, 0.8645211, 0.9341605],  # TBPTT
    [0.7232303000000001, 0.8160314, 0.8842865000000001, 0.9641953, 1.0422044000000001],  # SPTT-Window
], dtype=float)

BATCH_TIME_ERROR = np.array([
    [0.0005635, 0.0020244, 0.0011084, 4.38e-05, 0.0010719000000000002],  # BPTT
    [0.0007012, 0.0046779000000000005, 0.0007796, 0.0022053, 0.0025659000000000003],  # SPTT
    [0.0009593000000000001, 0.00024370000000000001, 0.0005393, 0.0021159, 0.0008999],  # SPTT-End
    [0.0022454999999999997, 0.0003403, 0.001391, 0.0036565, 0.003062],  # TBPTT
    [0.0008831, 0.0030886999999999998, 0.0018275, 0.0016681, 0.0003072],  # SPTT-Window
], dtype=float)

TOTAL_TIME_MEAN = np.array([
    [63.17028552833333, 87.77297019666666, 80.69969432333333, 97.83768431833333, 99.946403505],  # BPTT
    [61.575586236666666, 71.70465820833333, 76.42678905000001, 83.99228738333333, 91.63439091166666],  # SPTT
    [48.42714687833334, 56.18217075166667, 61.74454138, 67.98438624166667, 70.86542491],  # SPTT-End
    [54.952088368333335, 60.97559600833333, 66.99487239, 75.81663832666666, 78.65631693499999],  # TBPTT
    [58.356471176666666, 65.83926819666667, 71.35908568833334, 77.79839139666666, 87.75361056833334],  # SPTT-Window
], dtype=float)

TOTAL_TIME_ERROR = np.array([
    [11.969755981666667, 17.032270091666668, 11.163933445, 8.13381646, 0.12033657],  # BPTT
    [0.06395948166666666, 3.313256668333333, 0.07111089, 0.20115962666666667, 0.23404981833333335],  # SPTT
    [2.4457879666666664, 0.023937015, 0.05298011833333333, 0.20784803833333332, 3.7969348299999996],  # SPTT-End
    [0.189069435, 0.028652551666666668, 0.117121635, 3.968646591666667, 0.25782116499999996],  # TBPTT
    [3.517074745, 3.799533493333333, 4.534879606666666, 4.64928529, 0.02586860666666667],  # SPTT-Window
], dtype=float)

PEAK_MEMORY_MEAN = np.array([
    [3.7823783203125, 4.197712109375, 4.61314501953125, 5.0288328125, 5.444537109375],  # BPTT
    [4.98504716796875, 5.5514591796875, 6.11618876953125, 6.68416669921875, 7.24780341796875],  # SPTT
    [3.7878134765625, 4.2031880859375, 4.61863720703125, 5.03446650390625, 5.4506931640625],  # SPTT-End
    [1.28990078125, 1.393660546875, 1.497534765625, 1.601653125, 1.7057943359375],  # TBPTT
    [1.5949201171875, 1.73613076171875, 1.8782796875, 2.0187263671875, 2.15979482421875],  # SPTT-Window
], dtype=float)

PEAK_MEMORY_ERROR = np.array([
    [8.59375e-06, 7.03125e-06, 5.95703125e-06, 3.3203125e-06, 0.0],  # BPTT
    [0.0, 2.63671875e-06, 0.0, 0.0, 0.0],  # SPTT
    [2.5390625e-06, 0.0, 0.0, 0.0, 2.5390625e-06],  # SPTT-End
    [0.0, 0.0, 0.0, 0.0, 0.0],  # TBPTT
    [0.0, 0.0, 0.0, 0.0, 0.0],  # SPTT-Window
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
               ncol=5, frameon=False, handlelength=1.2, columnspacing=1.0,
               handletextpad=0.45, borderaxespad=0)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.17, wspace=0.30)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    base = SAVE_DIR / 'ag_news_efficiency_1x3'
    for suffix in ('png', 'jpg', 'pdf'):
        path = base.with_suffix('.' + suffix)
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(path)
    # Small preview for visual review; publication files remain at 600 dpi/vector.
    fig.savefig(base.with_name(base.name + '_preview.jpg'), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Validated {len(panels) * len(METHODS) * len(lengths)} bars; SPTT-Streaming excluded.')


if __name__ == '__main__':
    main()
