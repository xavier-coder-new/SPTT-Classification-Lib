# SPTT-Classification-Library

The repository includes the implementation of multiple classification tasks using SPTT and BPTT, which also contains various network architectures.

# Function Specification
| Function | Description |
| --- | --- |
| `reset_state(batch_size)` | Reset the states of hidden or cell |
| `reset_runtime_state(total_valid)` | Reset the states of accumulator and flg before starting the training for each chunk |
| `init_sptt_parameters() ` | It is only called during model initialization or when manually restarting the trajectory. |