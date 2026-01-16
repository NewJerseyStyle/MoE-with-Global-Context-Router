#!/bin/bash
# Run Token + Global Context MoE Experiments
# Tests different context and fusion methods

echo "========================================"
echo "Token + Global Context MoE Experiments"
echo "========================================"

cd "$(dirname "${BASH_SOURCE[0]}")"

# Experiment 1: Momentum + Gate (Our main approach)
echo ""
echo "[1/4] Testing Momentum + Gate..."
python evaluate_token_global.py \
    --context_method momentum \
    --fusion_method gate \
    --num_samples 100 \
    --output_dir ../results

# Experiment 2: THOR-style (context averaging)
echo ""
echo "[2/4] Testing THOR + Gate..."
python evaluate_token_global.py \
    --context_method thor \
    --fusion_method gate \
    --num_samples 100 \
    --output_dir ../results

# Experiment 3: Combined context
echo ""
echo "[3/4] Testing Combined + Gate..."
python evaluate_token_global.py \
    --context_method combined \
    --fusion_method gate \
    --num_samples 100 \
    --output_dir ../results

# Experiment 4: Simple addition (no learned gate)
echo ""
echo "[4/4] Testing Momentum + Add (no gate)..."
python evaluate_token_global.py \
    --context_method momentum \
    --fusion_method add \
    --num_samples 100 \
    --output_dir ../results

# Compare all results
echo ""
echo "========================================"
echo "Comparing all results..."
echo "========================================"
cd ..
python evaluate.py --compare --output_dir ./results

echo ""
echo "All experiments complete!"
