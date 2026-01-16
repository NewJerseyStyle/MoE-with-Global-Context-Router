#!/bin/bash
# Full Linear Baseline Experiment
# This script: 1) Merges models 2) Evaluates merged model 3) Evaluates individual experts

echo "========================================"
echo "MoE Routing Experiment - Linear Baseline"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Merge models with mergekit
echo "[Step 1/3] Merging models with mergekit linear..."
cd linear_baseline

if [ ! -f "merged_linear/config.json" ]; then
    echo "Running merge..."
    mergekit-yaml merge_config.yaml ./merged_linear --cuda --low-cpu-memory
    if [ $? -ne 0 ]; then
        echo "Merge failed!"
        exit 1
    fi
else
    echo "Merged model already exists, skipping merge."
fi

cd ..

# Step 2: Evaluate merged model
echo ""
echo "[Step 2/3] Evaluating merged linear model..."
python evaluate.py --model_path ./linear_baseline/merged_linear --method_name linear --num_samples 100

# Step 3: Evaluate individual experts for comparison
echo ""
echo "[Step 3/3] Evaluating individual expert models..."

echo "Evaluating base Qwen3-0.6B..."
python evaluate.py --model_path Qwen/Qwen3-0.6B --method_name base --num_samples 100

echo "Evaluating Math Expert..."
python evaluate.py --model_path suayptalha/Qwen3-0.6B-Math-Expert --method_name math_expert --num_samples 100

echo "Evaluating Code Expert..."
python evaluate.py --model_path suayptalha/Qwen3-0.6B-Code-Expert --method_name code_expert --num_samples 100

# Final comparison
echo ""
echo "========================================"
echo "Comparing all results..."
echo "========================================"
python evaluate.py --compare --output_dir ./results

echo ""
echo "Experiment complete! Results in ./results/"
