#!/bin/bash

# ============================================================

# SENTINEL Group Classification Runner

# First-time execution script

# ============================================================

echo "Starting SENTINEL group decoding pipeline..."

echo "-------------------------------------------"

PROJECT_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL"

cd $PROJECT_PATH || exit

PYTHON_SCRIPT="main.py"

SAVE_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/GroupDecoding"

DATA_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data"

BEHAV_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior"

TRIALINFO_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/trial_sentence_rank_info.pkl"

echo ""

echo "Running UNIFORM bootstrap (ALL TOI)"

echo ""

python3 $PYTHON_SCRIPT \
    --fPath "$DATA_PATH" \
    --bPath "$BEHAV_PATH" \
    --fileName "Data_sen_lepoch.pkl" \
    --trialInfo "$TRIALINFO_PATH" \
    --feature_mode uniform \
    --toi_mode all \
    --device mps \
    --epochs 40 \
    --patience 6 \
    --val_size 0.2 \
    --curve_dir "$SAVE_PATH/curves" \
    --verbose

echo ""

echo "Running CONTRAST bootstrap (ALL TOI)"

echo ""

python3 $PYTHON_SCRIPT \
    --fPath "$DATA_PATH" \
    --bPath "$BEHAV_PATH" \
    --fileName "Data_sen_lepoch.pkl" \
    --trialInfo "$TRIALINFO_PATH" \
    --feature_mode contrast \
    --toi_mode all \
    --device mps \
    --epochs 40 \
    --patience 6 \
    --val_size 0.2 \
    --curve_dir "$SAVE_PATH/curves" \
    --verbose

echo ""

echo "Running UNIFORM bootstrap (NON-BIO only)"

echo ""

python3 $PYTHON_SCRIPT \
    --fPath "$DATA_PATH" \
    --bPath "$BEHAV_PATH" \
    --fileName "Data_sen_lepoch.pkl" \
    --trialInfo "$TRIALINFO_PATH" \
    --feature_mode uniform \
    --toi_mode non_bio \
    --device mps \
    --epochs 40 \
    --patience 6 \
    --val_size 0.2 \
    --curve_dir "$SAVE_PATH/curves" \
    --verbose

echo ""

echo "Running CONTRAST bootstrap (NON-BIO only)"

echo ""

python3 $PYTHON_SCRIPT \
    --fPath "$DATA_PATH" \
    --bPath "$BEHAV_PATH" \
    --fileName "Data_sen_lepoch.pkl" \
    --trialInfo "$TRIALINFO_PATH" \
    --feature_mode contrast \
    --toi_mode non_bio \
    --device mps \
    --epochs 40 \
    --patience 6 \
    --val_size 0.2 \
    --curve_dir "$SAVE_PATH/curves" \
    --verbose

echo ""

echo "All decoding runs completed."

echo ""