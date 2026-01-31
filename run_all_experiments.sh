#!/bin/bash
# Run All MoE Routing Experiments
# This script runs all methods and compares results
#
# Methods:
#   1. Linear Baseline (mergekit static merge)
#   2. Model-Level Router (MoDEM-style)
#   3. Token + Global Context - Untrained (random router)
#   4. Token + Global Context - Trained (our main contribution)

echo "========================================"
echo "MoE Routing Experiment - Full Suite"
echo "========================================"
echo ""
echo "Methods to compare:"
echo "  1. Linear (static merge)"
echo "  2. MoDEM (model-level routing)"
echo "  3. Token+Global Untrained"
echo "  4. Token+Global Trained"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

NUM_SAMPLES=100
TRAIN_SAMPLES=500
TRAIN_EPOCHS=3

# ========================================
# Method 1: Linear Baseline (mergekit)
# ========================================
echo "[1/5] LINEAR BASELINE"
echo "========================================"

MERGED_MODEL_PATH="$SCRIPT_DIR/linear_baseline/merged_linear"

if [ ! -f "$MERGED_MODEL_PATH/config.json" ]; then
    echo "Merging models with mergekit..."
    cd "$SCRIPT_DIR/linear_baseline"
    mergekit-yaml merge_config.yaml "$MERGED_MODEL_PATH" --cuda --low-cpu-memory
    cd "$SCRIPT_DIR"

    # Verify merge succeeded
    if [ ! -f "$MERGED_MODEL_PATH/config.json" ]; then
        echo "ERROR: Merge failed! Skipping linear baseline evaluation."
        MERGED_MODEL_PATH=""
    fi
else
    echo "Merged model exists, skipping merge."
fi

if [ -n "$MERGED_MODEL_PATH" ] && [ -f "$MERGED_MODEL_PATH/config.json" ]; then
    python evaluate.py --model_path "$MERGED_MODEL_PATH" --method_name linear --num_samples $NUM_SAMPLES
else
    echo "Skipping linear baseline evaluation (no merged model)"
fi

# ========================================
# Method 2: Model-Level Router (MoDEM-style)
# ========================================
echo ""
echo "[2/5] MODEL-LEVEL ROUTER (MoDEM)"
echo "========================================"

cd "$SCRIPT_DIR/model_router"
python evaluate_model_router.py \
    --compute_centroids \
    --num_samples $NUM_SAMPLES \
    --output_dir "$SCRIPT_DIR/results"

cd "$SCRIPT_DIR"

# ========================================
# Method 3: Token + Global Context (Untrained)
# ========================================
echo ""
echo "[3/5] TOKEN + GLOBAL CONTEXT (Untrained)"
echo "========================================"

cd "$SCRIPT_DIR/token_global_router"

python evaluate_token_global.py \
    --context_method momentum \
    --fusion_method gate \
    --num_samples $NUM_SAMPLES \
    --output_dir "$SCRIPT_DIR/results"

cd "$SCRIPT_DIR"

# ========================================
# Method 4: Train Token + Global Router
# ========================================
echo ""
echo "[4/5] TRAINING TOKEN + GLOBAL ROUTER"
echo "========================================"

TRAINED_ROUTER_PATH="$SCRIPT_DIR/token_global_router/trained_router"

cd "$SCRIPT_DIR/token_global_router"

if [ ! -f "$TRAINED_ROUTER_PATH/final/config.json" ]; then
    echo "Training router..."
    python train_router.py \
        --num_epochs $TRAIN_EPOCHS \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_samples_per_domain $TRAIN_SAMPLES \
        --max_length 512 \
        --context_method momentum \
        --fusion_method gate \
        --load_balance_weight 0.01 \
        --output_dir "$TRAINED_ROUTER_PATH"
else
    echo "Trained router exists, skipping training."
fi

cd "$SCRIPT_DIR"

# ========================================
# Method 5: Token + Global Context (Trained)
# ========================================
echo ""
echo "[5/5] TOKEN + GLOBAL CONTEXT (Trained)"
echo "========================================"

cd "$SCRIPT_DIR/token_global_router"

if [ -f "$TRAINED_ROUTER_PATH/final/config.json" ]; then
    python evaluate_token_global.py \
        --context_method momentum \
        --fusion_method gate \
        --checkpoint_path "$TRAINED_ROUTER_PATH/final" \
        --num_samples $NUM_SAMPLES \
        --output_dir "$SCRIPT_DIR/results"
else
    echo "Skipping trained evaluation (training may have failed)"
fi

cd "$SCRIPT_DIR"

# ========================================
# Final Comparison
# ========================================
echo ""
echo "========================================"
echo "FINAL COMPARISON"
echo "========================================"

cd "$SCRIPT_DIR"
python evaluate.py --compare --output_dir "$SCRIPT_DIR/results"

echo ""
echo "========================================"
echo "All experiments complete!"
echo "========================================"
echo ""
echo "Results saved in: ./results/"
echo ""
echo "Summary of methods:"
echo "  - linear: Static weight merge (no routing)"
echo "  - model_router: MoDEM-style model-level routing"
echo "  - token_global_momentum_gate: Token+Global (untrained)"
echo "  - token_global_momentum_gate_trained: Token+Global (trained)"
echo "========================================"
