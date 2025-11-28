#!/bin/bash
# Complete workflow to run all anomaly detection models

DATA_PATH="./data/msl_fixed.csv"
BASE_OUTPUT="./results"

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    echo "Please update DATA_PATH variable in this script"
    exit 1
fi

echo "=========================================="
echo "Running Complete Anomaly Detection Pipeline"
echo "=========================================="
echo ""

# 1. LSTM Predictor + M2AD
echo "[1/7] Training LSTM Predictor + M2AD..."
python train_lstm_m2ad.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT/lstm_m2ad" \
    --window 120 \
    --hidden 128 \
    --n_layers 4 \
    --dropout 0.2 \
    --batch_size 64 \
    --epochs 30 \
    --patience 5 \
    --error_mode area \
    --area_l 2 \
    --ewma_alpha 0.3 \
    --gmm_components_max 3 \
    --target_fpr 0.01 \
    --min_event_len 5 \
    --dilate_radius 1

echo ""
echo "[1/7] LSTM+M2AD completed!"
echo ""

# 2. LSTM-AE + M2AD
echo "[2/7] Training LSTM-AE + M2AD..."
python train_ae_m2ad.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT/ae_m2ad" \
    --ae_window 100 \
    --hidden 128 \
    --n_layers 2 \
    --dropout 0.2 \
    --batch_size 256 \
    --epochs 30 \
    --patience 5 \
    --error_mode area \
    --area_l 2 \
    --ewma_alpha 0.3 \
    --gmm_components_max 3 \
    --target_fpr 0.01 \
    --min_event_len 5 \
    --dilate_radius 1

echo ""
echo "[2/7] LSTM-AE+M2AD completed!"
echo ""

# 3. LSTM-VAE + M2AD
echo "[3/7] Training LSTM-VAE + M2AD..."
python train_vae_m2ad.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT/vae_m2ad" \
    --ae_window 100 \
    --hidden 128 \
    --z_dim 64 \
    --n_layers 2 \
    --dropout 0.2 \
    --batch_size 64 \
    --epochs 30 \
    --patience 5 \
    --error_mode area \
    --area_l 2 \
    --ewma_alpha 0.3 \
    --gmm_components_max 3 \
    --target_fpr 0.01 \
    --min_event_len 5 \
    --dilate_radius 1

echo ""
echo "[3/7] LSTM-VAE+M2AD completed!"
echo ""

# 4. Standalone LSTM-AE
echo "[4/7] Training Standalone LSTM-AE..."
python train_ae_standalone.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT/ae_standalone" \
    --ae_window 100 \
    --hidden 128 \
    --n_layers 2 \
    --dropout 0.2 \
    --batch_size 256 \
    --epochs 30 \
    --patience 5 \
    --ewma_alpha 0.3 \
    --target_fpr 0.01 \
    --min_event_len 5 \
    --dilate_radius 1

echo ""
echo "[4/7] Standalone AE completed!"
echo ""

# 5. Standalone LSTM-VAE
echo "[5/7] Training Standalone LSTM-VAE..."
python train_vae_standalone.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT/vae_standalone" \
    --ae_window 100 \
    --hidden 128 \
    --z_dim 64 \
    --n_layers 2 \
    --dropout 0.2 \
    --batch_size 64 \
    --epochs 30 \
    --patience 5 \
    --ewma_alpha 0.3 \
    --target_fpr 0.01 \
    --min_event_len 5 \
    --dilate_radius 1

echo ""
echo "[5/7] Standalone VAE completed!"
echo ""

# 6. One-Class SVM (all kernels)
echo "[6/7] Training One-Class SVM (RBF, Linear, Poly + Ensemble)..."
python train_ocsvm.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT/ocsvm" \
    --rbf_nu 0.005 \
    --rbf_gamma auto \
    --linear_nu 0.001 \
    --poly_nu 0.005 \
    --poly_gamma auto \
    --poly_degree 2 \
    --poly_coef0 1.0 \
    --ensemble_weight_rbf 0.31 \
    --ensemble_weight_linear 0.36 \
    --ensemble_weight_poly 0.33 \
    --max_train_samples 250000 \
    --min_event_len 50 \
    --dilate_radius 1 \
    --merge_gap 3

echo ""
echo "[6/7] OCSVM completed!"
echo ""

# 7. Training RMSE Computation
echo "[7/7] Computing training RMSE for all models..."
python compute_train_rmse.py \
    --data_path "$DATA_PATH" \
    --output_dir "$BASE_OUTPUT" \
    --window 120 \
    --ae_window 100 \
    --hidden 128 \
    --n_layers 2 \
    --dropout 0.2 \
    --z_dim 64 \
    --batch_size 256 \
    --epochs 30 \
    --patience 5 \
    --pca_variance_ratio 0.90 \
    --ocsvm_weight_rbf 0.31 \
    --ocsvm_weight_linear 0.36 \
    --ocsvm_weight_poly 0.33

echo ""
echo "[7/7] RMSE computation completed!"
echo ""

# 8. Summary
echo "[8/8] Generating summary..."
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_OUTPUT/"
echo ""
echo "Key output files:"
echo "  - LSTM+M2AD metrics: $BASE_OUTPUT/lstm_m2ad/exact_metrics.csv"
echo "  - LSTM+M2AD metrics: $BASE_OUTPUT/lstm_m2ad/event_metrics.csv"
echo "  - LSTM-AE+M2AD metrics: $BASE_OUTPUT/ae_m2ad/ae_m2ad_exact_metrics.csv"
echo "  - LSTM-VAE+M2AD metrics: $BASE_OUTPUT/vae_m2ad/vae_m2ad_exact_metrics.csv"
echo "  - Standalone AE metrics: $BASE_OUTPUT/ae_standalone/ae_standalone_exact_metrics.csv"
echo "  - Standalone VAE metrics: $BASE_OUTPUT/vae_standalone/vae_standalone_exact_metrics.csv"
echo "  - OCSVM RBF metrics: $BASE_OUTPUT/ocsvm/ocsvm_rbf_metrics_exact.csv"
echo "  - OCSVM Linear metrics: $BASE_OUTPUT/ocsvm/ocsvm_linear_metrics_exact.csv"
echo "  - OCSVM Poly metrics: $BASE_OUTPUT/ocsvm/ocsvm_poly_metrics_exact.csv"
echo "  - OCSVM Ensemble metrics: $BASE_OUTPUT/ocsvm/ocsvm_ens3_MAJ_metrics_exact.csv"
echo "  - Training RMSE summary: $BASE_OUTPUT/train_rmse_summary.csv"
echo ""
echo "View RMSE summary:"
cat "$BASE_OUTPUT/train_rmse_summary.csv"
echo ""
