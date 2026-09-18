#  SPTT-Classification-Library

The repository includes the implementation of multiple classification tasks using SPTT and BPTT, which also contains various network architectures.

## Supported Tasks

Currently, this repository supports foundational frameworks for:

- **Text Classification**
- **Image Classification**

*☄️Future Plans:* Depending on updates and timeline availability, we plan to extend support to **Audio Classification** pipelines and **Time-Series Forecasting** tasks.

## Datasets

The repository includes built-in data loading and preprocessing pipelines for the following four benchmark datasets:

| **Dataset Name** | **Sequential-MNIST** | **LRA-CIFAR-10** | **AG News** | **LRA-Long Listops** |
| ---------------- | -------------------- | ---------------- | ----------- | -------------------- |
| **Type**         | Image                | Image            | Text        | Text                 |

### Data Configuration Notes

- **AG News:** Supports customizable sequence length control using a **padding and truncation** strategy. To specify the sequence length for AG News, you need to configure the following two parameters:
  - `max_length`: Specifies the maximum number of tokens.
  - `fixed_length`: Specifies the strict sequence length for alignment.
- **Storage:** Please download and place all required datasets into the `./datasets` directory before running the experiments.

## Environment Setup

You can automatically configure the required training environment using Conda:

```bash
conda env create -f sptt.yaml
```

## Running the Code

We provide three operational shell scripts in the `scripts/` directory to streamline your benchmarking:

- `demo_xxx.sh`: Runs a quick test on a specific dataset with customized parameters.
- `test.sh`: Serves as a unified entry point for standardized parameter execution.
- `total_xxx.sh`: Executes sequential testing across multiple datasets and parameter combinations automatically.

To initiate a baseline test, simply run:

```bash
./scripts/text_classification/demo_bptt.sh
```

## Time Statistics (Computational Profiling)

We provide built-in profiling tools to measure the runtime and FLOPs across different computational frameworks. To enable profiling for the BPTT and SPTT frameworks, please append the following arguments to your execution scripts:

- `--profile_bptt_compute`: Enables runtime and FLOPs profiling for the BPTT framework.
- `--profile_sptt_compute`: Enables runtime and FLOPs profiling for the SPTT framework.

## Visualization

We also provide a dedicated script to visualize the profiled runtime and FLOPs metrics. For detailed implementation logic, please refer to `tools/plot_time_grid.py`. To execute the visualization directly, you can run the provided shell script:

```bash
./scripts/plot_time.sh
```

## Ablation

To investigate whether the performance gains of SPTT stem solely from implicit regularization, we provide ablation experiments for gradient clipping and low-rank BPTT. Specifically, the low-rank BPTT approach first computes the full gradient normally and then applies low-rank decomposition to the complete gradient for parameter updates.

To enable gradient clipping, please use the --use_clip flag and specify the --clip_norm value. To apply low-rank decomposition to the BPTT gradient, please use the --bptt_low_rank flag. We have provided a sample run script that you can execute directly as follows:

```bash
./scripts/image_classification/total_gradient.sh 
```

## Investigating Temporal Continuity

To study how temporal continuity affects BPTT and SPTT, we add a controlled inheritance-switch experiment. Two flags, `--inherit_sptt` and `--inherit_bptt`, determine whether the current gradient of SPTT or BPTT is updated from the previous iteration’s gradient.

The goal is to examine SPTT’s *non-inheritance* behavior and BPTT’s *inheritance* behavior:

- For SPTT, set `inherit_sptt=0`. The three components that form the gradient—`X_matrix`, `Sigma_matrix`, and `Delta_matrix`—are then re-initialized from scratch at every backward pass.
- For BPTT, set `inherit_bptt=1`. BPTT first uses the low-rank factorization from earlier experiments, then applies the same raw-average form as SPTT, combining gradient information from the previous iteration and the current one.

See `exp_image_classification.py` for the implementation.

Ready-to-run scripts:

```bash
# For BPTT
./scripts/image_classification/total_bptt_inherit.sh
# For SPTT
./scripts/image_classification/total_sptt_inherit.sh
```

## End-to-End Statistics

Building on the earlier kernel-isolated timing and FLOPs measurements for recurrent gradient computation, we further report end-to-end time and GPU memory for the different computation frameworks. See `exp_text_classification.py`.

We also provide a memory-reuse variant of SPTT. In the original implementation (`SpttLSTM.py`), SPTT-Hybrid explicitly stores all activations and error signals and then slides over them, which inflates memory use. The reuse version keeps a fixed memory budget: once that budget is reached, the program performs one SPTT update and immediately frees the buffer. This substantially reduces peak memory. See `SpttLSTM_Streaming.py`.

End-to-end statistics can be collected with:

```bash
./scripts/text_classification/total_time_flops.sh
```

🍔**Note:** you must specify the model name yourself. For `SpttLSTM_Streaming`, set `--slide_window_nums` to control the size of the reused memory partition. For `SpttLSTM`, this argument defaults to `1`.

## Key Parameter Configurations

- **Sequence Truncation:** The dataset supports both full and truncated sequence processing. To process the complete sequence, set `truncate_num=1`. To evaluate TBPTT or SPTT-Window, set `truncate_num` to your desired time-chunk size.
- **`slide_window_nums`:** Specifies the incremental update frequency for the SPTT-Hybrid mode. Note that this parameter has no effect on the BPTT framework.
- **`length_mode`:** Supports two sequence alignment strategies: `pad` and `repeat`. The `pad` mode applies standard zero-padding. The `repeat` mode is specifically designed for the AG News dataset, where it continuously replicates and truncates the sequence to reach the target length.
- **`num_layers`:** Specifies the depth of the network. When benchmarking the fundamental performance and characteristics of the computational frameworks, we highly recommend setting this to `1` for better controllability. In our empirical tests, increasing the number of layers did not yield significant performance improvements but substantially extended the training time. Furthermore, on simpler datasets, deeper networks actually degraded BPTT's performance, whereas SPTT consistently maintained its stability.