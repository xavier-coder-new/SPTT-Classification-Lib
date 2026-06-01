# plot_sptt_compute_metrics.py

from pathlib import Path
from tools.sptt_compute_profiler import plot_sptt_compute_metrics

summary_csv = Path(
    "compute_metrics"
    "/image_classification"
    "/SpttLSTM"
    "/sequential_mnist"
    "/seed_2024"
    "/krank_10_Trun_1_Slide_4"
    "/sptt_compute_epoch_summary.csv"
)

plot_sptt_compute_metrics(
    summary_csv,
    save_path=summary_csv.parent / "sptt_compute_metrics.svg",
    title="SPTT-LSTM",
)