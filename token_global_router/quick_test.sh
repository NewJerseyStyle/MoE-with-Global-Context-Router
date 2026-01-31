#!/bin/bash
# Quick test with fewer samples and experts

echo "Quick Test: Token + Global MoE (2 experts, 10 samples)"

cd "$(dirname "${BASH_SOURCE[0]}")"

python evaluate_token_global.py \
    --context_method momentum \
    --fusion_method gate \
    --num_samples 10 \
    --experts base math \
    --benchmarks gsm8k \
    --output_dir ../results

echo "Quick test complete!"
