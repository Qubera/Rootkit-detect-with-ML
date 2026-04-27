# Syscall ML Single-File CLI

This project is now intentionally simplified into one Python file:

- `syscall_ml.py`

The script provides simple terminal управление through commands:

- `collect`
- `c`
- `check`
- `k`
- `train`
- `t`
- `predict`
- `p`
- `realtime`
- `r`

It is designed for:

- collecting syscall traces on Linux
- training a syscall-based ML model
- running inference on saved syscall traces
- performing near-real-time analysis from a growing syscall CSV file

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Input data format

### Training CSV

Required columns:

- `pid`
- `syscall`
- `label`

Example:

```csv
pid,syscall,label
101,open,0
101,read,0
101,write,0
202,execve,1
202,open,1
202,write,1
```

Rules:

- one `pid` must have one label
- rows for a process must stay in syscall order
- `0` usually means benign
- `1` usually means suspicious or malicious-like

### Inference CSV

For `predict` and `realtime`, `label` is optional.

Example:

```csv
pid,syscall
301,open
301,read
301,write
```

## 3. Commands

### Collect

The `collect` command is the built-in Linux syscall collector.

It uses `strace` to run a target program, parses the resulting syscall trace, and appends the parsed events to a CSV file.

Example:

```bash
python syscall_ml.py c -b -- python app.py
```

Another example:

```bash
python syscall_ml.py c -s -- ./sample_program
```

Fastest basic collection:

```bash
python syscall_ml.py c -b
```

If you run `c -b` without a command, the script automatically traces a built-in benign set of Linux commands:

- `ls -la`
- `cat /etc/hosts`
- `uname -a`
- `id`
- `whoami`
- `ps aux`

What `collect` does:

- checks that the system is Linux
- checks that `strace` is installed
- runs the target command under `strace`
- or runs a built-in benign command set if no command was provided
- saves the raw trace log
- parses `pid` and `syscall`
- optionally assigns a label
- appends parsed rows to your dataset CSV

Main arguments:

- `--output-csv`
- `--raw-dir`
- `--label`
- `--benign` / `--suspicious`
- target command after `--`

Notes:

- `collect` works only on Linux
- it does not require splitting the project into multiple files
- collected CSV rows may contain extra metadata columns such as `collected_at` and `source_command`

### Train

Basic command:

```bash
python syscall_ml.py t --csv syscalls.csv --output-dir artifacts
```

Extended experiment:

```bash
python syscall_ml.py t --csv syscalls.csv --output-dir artifacts --window-sizes 3,5,10 --top-bigrams-grid 20,50,100 --cv-splits 3
```

What `train` does:

- loads and validates the dataset
- splits train and test by `pid`
- builds sliding windows
- extracts features
- compares `RandomForest` and `XGBoost`
- compares several `window_size` and `top_bigrams` values
- trains the best model
- evaluates on a real holdout split
- saves metrics, plots, predictions, and a markdown report

Main arguments:

- `--csv`
- `--output-dir`
- `--window-sizes`
- `--top-bigrams-grid`
- `--test-size`
- `--min-bigram-freq`
- `--cv-splits`

### Predict

Run inference on a saved CSV:

```bash
python syscall_ml.py p --csv new_syscalls.csv --artifacts-dir artifacts --output-dir predictions
```

What `predict` does:

- loads the trained model
- loads the fitted feature extractor
- converts the new trace into windows
- predicts suspicious probability
- saves:
  - `predictions/window_predictions.csv`
  - `predictions/pid_predictions.csv`
  - `predictions/suspicious_pids.csv`

You can tune suspicious PID reporting with:

- `--threshold`

### Realtime

Run near-real-time analysis from a growing CSV file:

```bash
python syscall_ml.py r --csv live_syscalls.csv --artifacts-dir artifacts --output-dir live_output --interval 5 --threshold 0.7
```

What `realtime` does:

- watches the CSV file
- notices when the file changes
- reloads the current data
- runs inference again
- updates live results
- prints suspicious PIDs in the terminal

Main arguments:

- `--csv`
- `--artifacts-dir`
- `--output-dir`
- `--interval`
- `--threshold`
- `--once`

## 4. Output files

### Training outputs

- `artifacts/model.pkl`
- `artifacts/feature_columns.pkl`
- `artifacts/feature_extractor.pkl`
- `artifacts/dataset_summary.json`
- `artifacts/training_config.json`
- `artifacts/cv_results.csv`
- `artifacts/cv_results.md`
- `artifacts/cv_f1_comparison.png`
- `artifacts/holdout_predictions.csv`
- `artifacts/confusion_matrix.png`
- `artifacts/shap_summary.png`
- `artifacts/experiment_report.md`
- `artifacts/metrics.json`

### Predict outputs

- `predictions/window_predictions.csv`
- `predictions/pid_predictions.csv`
- `predictions/suspicious_pids.csv`

### Collect outputs

- `collector_output/strace_*.log`
- `collector_output/last_collect_summary.json`
- updated `syscalls.csv`

### Realtime outputs

- `live_output/live_window_predictions.csv`
- `live_output/live_pid_predictions.csv`
- `live_output/live_suspicious_pids.csv`
- `live_output/live_status.json`

## 5. Metrics

The script reports:

- `accuracy`
- `balanced_accuracy`
- `precision`
- `recall`
- `f1-score`
- `roc-auc`
- `confusion matrix`
- `classification report`

## 6. Quick Diamorphine demo

If you want the easiest possible demo flow, use the built-in `demo` command.

On a clean Linux system:

```bash
python syscall_ml.py demo benign
```

After you manually install/load Diamorphine:

```bash
python syscall_ml.py demo infected
```

Train the model:

```bash
python syscall_ml.py demo train
```

Then show detection on the current machine state:

```bash
python syscall_ml.py demo detect
```

You can print the built-in walkthrough any time:

```bash
python syscall_ml.py demo guide
```

This flow reuses the same simple Linux commands before and after infection, which makes the before/after comparison easy to explain during a demo.

## 7. Typical workflow

### Step 1

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 2

Collect or prepare:

```text
syscalls.csv
```

Example collection:

```bash
python syscall_ml.py c -b
```

### Step 3

Check the dataset before training:

```bash
python syscall_ml.py check
```

or short:

```bash
python syscall_ml.py k
```

### Step 4

Train:

```bash
python syscall_ml.py t --csv syscalls.csv --output-dir artifacts
```

### Step 5

Review:

- `artifacts/experiment_report.md`
- `artifacts/cv_results.csv`
- `artifacts/confusion_matrix.png`
- `artifacts/shap_summary.png`

### Step 6

Predict on new traces:

```bash
python syscall_ml.py p --csv new_syscalls.csv --artifacts-dir artifacts --output-dir predictions
```

### Step 7

Start live monitoring:

```bash
python syscall_ml.py r --csv live_syscalls.csv --artifacts-dir artifacts --output-dir live_output --interval 5
```

## 8. End-to-end system workflow

The full workflow is now:

1. `collect` syscall traces from Linux programs
2. build a labeled `syscalls.csv`
3. `train` the ML model
4. `predict` on saved traces
5. `realtime` monitor a growing syscall CSV

This makes the project much closer to a complete diploma system instead of a standalone offline ML script.

Minimal workflow:

```bash
python syscall_ml.py c -b
python syscall_ml.py k
python syscall_ml.py t
python syscall_ml.py p
python syscall_ml.py r --csv live_syscalls.csv
```

## 9. Important note about real-time mode

This is near-real-time monitoring, not hard real-time.

The script expects a CSV file that is continuously updated with syscall events. It re-runs analysis whenever that file changes.

That makes it suitable for:

- demos
- diploma defense
- laboratory monitoring
- experiment replay

## 10. Current limitation

The real-time mode does not capture syscalls by itself. It analyzes a syscall log file that is being updated by some external collection process.

So the workflow is:

1. collect syscall events into a CSV using `collect` or another Linux source
2. point `syscall_ml.py realtime` at that CSV
3. watch live predictions update in the terminal and output files
