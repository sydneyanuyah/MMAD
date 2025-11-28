# Quick Start Guide

## Installation

```bash
# Clone or download the repository
cd anomaly-detection

# Install dependencies
pip install -r requirements.txt
```

## Prepare Your Data

Ensure your CSV file has the following structure:
- `chan_id`: Channel identifier
- `subset`: 'train' or 'test'
- `time_step`: Time index
- `feature_0`, `feature_1`, ..., `feature_N`: Feature columns
- `anomaly_label`: Binary labels (0 or 1)

## Quick Run - All Models

```bash
# Make the script executable
chmod +x run_all.sh

# Update DATA_PATH in run_all.sh to point to your data
# Then run:
./run_all.sh
```

This will run all models and save results to `./results/`

## Run Individual Models

### 1. LSTM Predictor + M2AD (Recommended)

```bash
python train_lstm_m2ad.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/lstm_m2ad \
    --window 120 \
    --epochs 30
```

**Key parameters to tune:**
- `--window`: Sequence length (default: 120)
- `--hidden`: Hidden layer size (default: 128)
- `--n_layers`: Number of LSTM layers (default: 4)
- `--error_mode`: 'point' or 'area' (default: area)
- `--target_fpr`: Target false positive rate (default: 0.01)

### 2. One-Class SVM

```bash
python train_ocsvm.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/ocsvm \
    --rbf_nu 0.005 \
    --linear_nu 0.001 \
    --poly_nu 0.005 \
    --min_event_len 50
```

**Key parameters to tune:**
- `--rbf_nu`, `--linear_nu`, `--poly_nu`: Nu parameters for each kernel
- `--min_event_len`: Minimum anomaly event length (default: 50)
- `--max_train_samples`: Max training samples (default: 250000)

### 3. Training RMSE Analysis

```bash
python compute_train_rmse.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results \
    --reuse_saved
```

**Key parameters:**
- `--reuse_saved`: Reuse previously computed reconstructions
- `--pca_variance_ratio`: PCA components ratio (default: 0.90)

## Understanding Results

### Metrics Files

Each model generates two metrics files:
- `*_metrics_exact.csv`: Point-wise metrics (precision, recall, F1, F0.5)
- `*_metrics_event.csv`: Event-wise metrics (per anomaly event)

### Key Metrics

- **F1**: Harmonic mean of precision and recall
- **F0.5**: Weighted F-score favoring precision
- **tp, fp, fn**: True/False positives, False negatives

### RMSE Files

- `train_rmse_per_channel.csv`: RMSE for each model and channel
- `train_rmse_summary.csv`: Aggregated statistics by model

## Configuration Differences

### LSTM Models (Predictor, AE, VAE)
- Window: 120 (predictor) or 100 (AE/VAE)
- Min event length: 5
- Error mode: area (uses local integration)

### OCSVM Models
- Min event length: **50** (different from LSTM!)
- Uses StandardScaler normalization
- Max training samples: 250,000 (for efficiency)

### RMSE Computation
- OCSVM: Uses KernelPCA (RBF/Poly) or PCA (Linear) for reconstruction
- Weighted ensemble: 0.31*RBF + 0.36*Linear + 0.33*Poly

## Common Issues

### 1. Out of Memory
- Reduce `--batch_size` (try 32 or 16)
- Reduce `--max_train_samples` for OCSVM
- Use CPU instead: `--device cpu`

### 2. Slow Training
- Use GPU: `--device cuda` (if available)
- Reduce `--epochs` or increase `--patience`
- For OCSVM, reduce `--max_train_samples`

### 3. Poor Results
- Adjust `--target_fpr` (try 0.001 to 0.05)
- Change `--min_event_len` (affects event merging)
- Tune nu parameters for OCSVM (typically 0.001 to 0.01)

## Output Structure

```
results/
├── lstm_m2ad/
│   ├── config.json
│   ├── exact_metrics.csv
│   ├── event_metrics.csv
│   └── m2ad_fit_params.json
├── ocsvm/
│   ├── ocsvm_config.json
│   ├── ocsvm_rbf_metrics_exact.csv
│   ├── ocsvm_rbf_metrics_event.csv
│   ├── ocsvm_linear_metrics_exact.csv
│   ├── ocsvm_linear_metrics_event.csv
│   ├── ocsvm_poly_metrics_exact.csv
│   ├── ocsvm_poly_metrics_event.csv
│   ├── ocsvm_ens3_MAJ_metrics_exact.csv
│   └── ocsvm_ens3_MAJ_metrics_event.csv
├── train_rmse_per_channel.csv
└── train_rmse_summary.csv
```


## Getting Help

For detailed parameter descriptions:
```bash
python train_lstm_m2ad.py --help
python train_ocsvm.py --help
python compute_train_rmse.py --help
```

See `README.md` for complete documentation.
