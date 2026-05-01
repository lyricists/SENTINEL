#!/bin/bash

# ============================================================
# SENTINEL Group Classification Runner
# DeepConvNet version
# ============================================================

echo "Starting SENTINEL group decoding pipeline with DeepConvNet..."
echo "-------------------------------------------"

PROJECT_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL"

cd "$PROJECT_PATH" || exit

PYTHON_SCRIPT="main.py"

SAVE_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/GroupDecoding/DeepConvNet"

DATA_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data"

BEHAV_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior"

TRIALINFO_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/trial_sentence_rank_info_dl.pkl"

COMMON_ARGS=(
    --fPath "$DATA_PATH"
    --bPath "$BEHAV_PATH"
    --fileName "Data_sen_lepoch.pkl"
    --trialInfo "$TRIALINFO_PATH"
    --model_type deepconvnet
    --device mps
    --val_size 0.1
    --curve_dir "$SAVE_PATH/curves"
    --chName None
    --early_stop_metric val_balanced_accuracy
    --verbose
)

mkdir -p "$SAVE_PATH/curves"

echo ""
echo "Running UNIFORM bootstrap (ALL TOI)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode uniform \
    --toi_mode all \
    --epochs 60 \
    --patience 15

echo ""
echo "Running CONTRAST bootstrap (ALL TOI)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode contrast \
    --toi_mode all \
    --epochs 60 \
    --patience 15

echo ""
echo "Running sentence response (ALL TOI)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode sentence_response \
    --toi_mode all \
    --epochs 60 \
    --patience 15

echo ""
echo "Running UNIFORM bootstrap (NON-BIO only)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode uniform \
    --toi_mode non_bio \
    --epochs 60 \
    --patience 15

echo ""
echo "Running CONTRAST bootstrap (NON-BIO only)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode contrast \
    --toi_mode non_bio \
    --epochs 60 \
    --patience 15

echo ""
echo "Running sentence response (NON-BIO only)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode sentence_response \
    --toi_mode non_bio \
    --epochs 60 \
    --patience 15

echo ""
echo "All EEGNet decoding runs completed."
echo ""