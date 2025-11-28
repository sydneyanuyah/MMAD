# Anomaly Detection Framework

This repository contains a modular implementation of various anomaly detection methods for multivariate time series data, including all 9 models:

**Deep Learning Models:**
1. LSTM Predictor + M2AD
2. LSTM Autoencoder + M2AD
3. LSTM VAE + M2AD
4. LSTM Autoencoder (standalone)
5. LSTM VAE (standalone)

**One-Class SVM Models:**
6. OCSVM with RBF kernel
7. OCSVM with Linear kernel
8. OCSVM with Polynomial kernel
9. OCSVM 3-Model Ensemble (Majority Vote)

## Project Structure

```
├── config.py                  # Configuration and argument parsing
├── data_loader.py             # Data loading and preprocessing
├── models.py                  # Neural network model definitions
├── trainers.py                # Training and inference functions
├── m2ad_scorer.py             # M2AD scoring implementation
├── utils.py                   # Utility functions (metrics, post-processing)
├── train_lstm_m2ad.py         # LSTM Predictor + M2AD
├── train_ae_m2ad.py           # LSTM-AE + M2AD
├── train_vae_m2ad.py          # LSTM-VAE + M2AD
├── train_ae_standalone.py     # Standalone LSTM-AE
├── train_vae_standalone.py    # Standalone LSTM-VAE
├── train_ocsvm.py             # OCSVM (RBF, Linear, Poly + Ensemble)
├── compute_train_rmse.py      # Training RMSE computation
├── run_all.sh                 # Batch script to run all models
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── QUICK_START.md             # Quick start guide
```

## Installation

```bash
pip install numpy pandas scikit-learn scipy torch tqdm
```

## Usage

### Basic Training

Train LSTM Predictor with M2AD scorer:

```bash
python train_lstm_m2ad.py \
    --data_path /path/to/msl_fixed.csv \
    --output_dir ./results \
    --window 120 \
    --epochs 30 \
    --batch_size 64
```

### Train One-Class SVM with Multiple Kernels

```bash
python train_ocsvm.py \
    --data_path /path/to/msl_fixed.csv \
    --output_dir ./results/ocsvm \
    --rbf_nu 0.005 \
    --linear_nu 0.001 \
    --poly_nu 0.005 \
    --poly_degree 2 \
    --min_event_len 50
```

This will train three OCSVM models (RBF, Linear, Poly kernels) and create a 3-model majority vote ensemble.

### Compute Training RMSE

Compute training reconstruction RMSE for all models including OCSVM (using KernelPCA/PCA):

```bash
python compute_train_rmse.py \
    --data_path /path/to/msl_fixed.csv \
    --output_dir ./results \
    --window 120 \
    --ae_window 100 \
    --pca_variance_ratio 0.90 \
    --reuse_saved
```

The `--reuse_saved` flag will reuse previously computed reconstructions if available.

### Configuration Options

#### Data Parameters
- `--data_path`: Path to input CSV file (required)
- `--output_dir`: Output directory for results (default: 'anom_exp_latest')
- `--features_prefix`: Prefix for feature columns (default: 'feature_')
- `--label_col`: Label column name (default: 'anomaly_label')
- `--chan_col`: Channel ID column (default: 'chan_id')
- `--subset_col`: Train/test subset column (default: 'subset')
- `--time_col`: Time step column (default: 'time_step')

#### Model Architecture
- `--window`: Window size for LSTM predictor (default: 120)
- `--ae_window`: Window size for AE/VAE (default: 100)
- `--hidden`: Hidden dimension size (default: 128)
- `--n_layers`: Number of LSTM layers (default: 4)
- `--dropout`: Dropout rate (default: 0.2)
- `--z_dim`: Latent dimension for VAE (default: 64)

#### Training Parameters
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 30)
- `--patience`: Early stopping patience (default: 5)
- `--lr`: Learning rate (default: 0.001)

#### M2AD Parameters
- `--error_mode`: Error computation mode, 'point' or 'area' (default: 'area')
- `--area_l`: Half-window for area integral (default: 2)
- `--ewma_alpha`: EWMA smoothing alpha (default: 0.3)
- `--gmm_components_max`: Max GMM components (default: 3)
- `--target_fpr`: Target false positive rate (default: 0.01)
- `--min_p`: Minimum p-value (default: 1e-12)

#### Post-Processing
- `--min_event_len`: Minimum event length (default: 5 for LSTM models, 50 for OCSVM)
- `--dilate_radius`: Dilation radius for events (default: 1)

#### OCSVM-Specific Parameters
- `--rbf_nu`: Nu parameter for RBF kernel (default: 0.005)
- `--rbf_gamma`: Gamma parameter for RBF (default: 'auto')
- `--linear_nu`: Nu parameter for linear kernel (default: 0.001)
- `--poly_nu`: Nu parameter for poly kernel (default: 0.005)
- `--poly_degree`: Degree for polynomial kernel (default: 2)
- `--poly_coef0`: Coef0 for polynomial kernel (default: 1.0)
- `--ensemble_weight_rbf`: Weight for RBF in ensemble (default: 0.31)
- `--ensemble_weight_linear`: Weight for linear in ensemble (default: 0.36)
- `--ensemble_weight_poly`: Weight for poly in ensemble (default: 0.33)
- `--max_train_samples`: Max samples for training (default: 250000)
- `--merge_gap`: Min gap to merge events (default: 3)

#### RMSE Computation Parameters
- `--pca_variance_ratio`: Variance ratio for PCA components (default: 0.90)
- `--ocsvm_weight_rbf`: Weight for RBF reconstruction (default: 0.31)
- `--ocsvm_weight_linear`: Weight for Linear reconstruction (default: 0.36)
- `--ocsvm_weight_poly`: Weight for Poly reconstruction (default: 0.33)
- `--reuse_saved`: Reuse saved reconstruction files if available

#### Runtime
- `--num_workers`: Data loading workers (default: 0)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use: 'auto', 'cuda', or 'cpu' (default: 'auto')

## Output Files

The training scripts generate the following outputs in the specified output directory:

### LSTM+M2AD / AE+M2AD / VAE+M2AD
1. **config.json**: Complete configuration used for the run
2. **exact_metrics.csv**: Point-wise anomaly detection metrics per channel
3. **event_metrics.csv**: Event-wise anomaly detection metrics per channel
4. **m2ad_fit_params.json**: M2AD fitting parameters per channel

### OCSVM
1. **ocsvm_config.json**: OCSVM configuration
2. **ocsvm_{kernel}_metrics_exact.csv**: Point-wise metrics for each kernel (rbf, linear, poly)
3. **ocsvm_{kernel}_metrics_event.csv**: Event-wise metrics for each kernel
4. **ocsvm_{kernel}_{channel}_scores.csv**: Per-channel scores and predictions
5. **ocsvm_ens3_MAJ_metrics_exact.csv**: Ensemble exact metrics
6. **ocsvm_ens3_MAJ_metrics_event.csv**: Ensemble event metrics

### Training RMSE
1. **train_rmse_per_channel.csv**: RMSE for each model and channel
2. **train_rmse_summary.csv**: Aggregated RMSE statistics by model
3. **{model}_train_recon_{channel}.npy**: Saved reconstruction arrays

### Metrics Columns

Both metrics files contain:
- `model`: Model name
- `dataset`: Channel ID
- `precision`: Precision score
- `recall`: Recall score
- `F1`: F1 score
- `F0.5`: F0.5 score
- `tp`: True positives
- `fp`: False positives
- `fn`: False negatives

For RMSE files:
- `channel`: Channel ID
- `model`: Model name
- `rmse_train`: Training RMSE value

## Example Workflow

```bash
# Run all 9 models at once
./run_all.sh

# Or run individual models:

# 1. LSTM Predictor + M2AD
python train_lstm_m2ad.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/lstm_m2ad \
    --window 120 \
    --hidden 128 \
    --epochs 30

# 2. LSTM-AE + M2AD
python train_ae_m2ad.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/ae_m2ad \
    --ae_window 100 \
    --hidden 128 \
    --epochs 30

# 3. LSTM-VAE + M2AD
python train_vae_m2ad.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/vae_m2ad \
    --ae_window 100 \
    --z_dim 64 \
    --epochs 30

# 4. Standalone LSTM-AE
python train_ae_standalone.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/ae_standalone \
    --ae_window 100 \
    --epochs 30

# 5. Standalone LSTM-VAE
python train_vae_standalone.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/vae_standalone \
    --ae_window 100 \
    --z_dim 64 \
    --epochs 30

# 6-9. One-Class SVM (all kernels + ensemble)
python train_ocsvm.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results/ocsvm \
    --rbf_nu 0.005 \
    --linear_nu 0.001 \
    --poly_nu 0.005 \
    --min_event_len 50

# 10. Compute training RMSE for all models
python compute_train_rmse.py \
    --data_path ./data/msl_fixed.csv \
    --output_dir ./results \
    --window 120 \
    --ae_window 100 \
    --pca_variance_ratio 0.90

# 11. Compare results
cat ./results/train_rmse_summary.csv
```

## Data Format

The input CSV should have the following structure:

- **Channel ID column** (default: `chan_id`): Identifier for each time series
- **Subset column** (default: `subset`): 'train' or 'test' indicator
- **Time step column** (default: `time_step`): Time index
- **Feature columns** (default prefix: `feature_`): Multivariate features (e.g., feature_0, feature_1, ...)
- **Label column** (default: `anomaly_label`): Binary anomaly labels (0/1)

Example:
```
chan_id,subset,time_step,feature_0,feature_1,...,feature_54,anomaly_label
A-1,train,0,0.123,0.456,...,0.789,0
A-1,train,1,0.124,0.457,...,0.790,0
A-1,test,0,0.125,0.458,...,0.791,0
...
```

## Key Features

1. **Modular Design**: Separate modules for models, training, scoring, and utilities
2. **Flexible Configuration**: Comprehensive command-line arguments with sensible defaults
3. **Reproducibility**: Seed control and configuration logging
4. **Event-Level Metrics**: Both point-wise and event-wise evaluation
5. **Post-Processing**: Canonical event dilation and minimum length filtering
6. **Early Stopping**: Automatic early stopping to prevent overfitting
7. **GPU Support**: Automatic GPU detection and usage
8. **Multiple OCSVM Kernels**: RBF, Linear, Poly + Majority Vote Ensemble
9. **Reconstruction Analysis**: KernelPCA/PCA-based reconstruction for OCSVM
10. **Training RMSE Computation**: Comprehensive reconstruction quality evaluation

## Important Notes

### OCSVM vs LSTM Configuration Differences

**OCSVM uses different post-processing defaults:**
- `min_event_len=50` (vs 5 for LSTM models)
- This matches your notebook's OCSVM tuning configuration
- Use `--min_event_len` flag to adjust

**OCSVM Reconstruction Method:**
- RBF & Poly: KernelPCA with `fit_inverse_transform=True`
- Linear: Standard PCA (faster and more stable)
- Ensemble: Weighted combination (0.31*RBF + 0.36*Linear + 0.33*Poly)
- Components: 90% of features by default (use `--pca_variance_ratio` to adjust)

## Notes

- All hyperparameters match the original notebook implementation
- The code preserves the exact logic and variable names from your notebook
- Results should be identical to the notebook when using the same seed and data
- The framework supports easy extension to additional models or scorers
