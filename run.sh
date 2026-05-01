#!/bin/bash

# ============================================================
# SENTINEL Group Classification Runner
# EEGNet version
# ============================================================

echo "Starting SENTINEL group decoding pipeline with EEGNet..."
echo "-------------------------------------------"

PROJECT_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL"

cd "$PROJECT_PATH" || exit

PYTHON_SCRIPT="main.py"

SAVE_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/GroupDecoding/EEGNet"

DATA_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data"

BEHAV_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior"

TRIALINFO_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/trial_sentence_rank_info.pkl"

COMMON_ARGS=(
    --fPath "$DATA_PATH"
    --bPath "$BEHAV_PATH"
    --fileName "Data_sen_lepoch.pkl"
    --trialInfo "$TRIALINFO_PATH"
    --model_name eegnet
    --device mps
    --val_size 0.2
    --curve_dir "$SAVE_PATH/curves"
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
    --epochs 40 \
    --patience 6

echo ""
echo "Running CONTRAST bootstrap (ALL TOI)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode contrast \
    --toi_mode all \
    --epochs 40 \
    --patience 6

echo ""
echo "Running sentence response (ALL TOI)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode sentence_response \
    --toi_mode all \
    --epochs 100 \
    --patience 50

echo ""
echo "Running UNIFORM bootstrap (NON-BIO only)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode uniform \
    --toi_mode non_bio \
    --epochs 40 \
    --patience 6

echo ""
echo "Running CONTRAST bootstrap (NON-BIO only)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode contrast \
    --toi_mode non_bio \
    --epochs 40 \
    --patience 6

echo ""
echo "Running sentence response (NON-BIO only)"
echo ""

python3 "$PYTHON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --feature_mode sentence_response \
    --toi_mode non_bio \
    --epochs 100 \
    --patience 50

echo ""
echo "All EEGNet decoding runs completed."
echo ""