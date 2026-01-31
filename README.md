# MoE Routing Experiment

Comparing different MoE routing strategies on Qwen3-0.6B fine-tuned expert models.

## Research Question

Does token-level routing with global context outperform:
1. Static linear merge (no routing)
2. Model-level routing (MoDEM-style)

## Expert Models

| Model | Domain |
|-------|--------|
| Qwen/Qwen3-0.6B | Base |
| suayptalha/Qwen3-0.6B-Medical-Expert | Medical |
| suayptalha/Qwen3-0.6B-Code-Expert | Code |
| suayptalha/Qwen3-0.6B-IF-Expert | Instruction Following |
| suayptalha/Qwen3-0.6B-Math-Expert | Math |

## Methods Compared

| Method | Granularity | Training | Description |
|--------|-------------|----------|-------------|
| **Linear** | None | No | Static weight merge (mergekit) |
| **MoDEM** | Input-level | No | Model-level routing via embedding similarity |
| **Token+Global (Untrained)** | Token-level | No | Random router + global context |
| **Token+Global (Trained)** | Token-level | Yes | Trained router + global context (ours) |

## Quick Start

```bash
# Install dependencies
pip install mergekit transformers datasets torch tqdm

# Run ALL experiments (includes training)
bash run_all_experiments.sh
```

This will:
1. Merge models with mergekit (Linear baseline)
2. Evaluate MoDEM-style model-level routing
3. Evaluate Token+Global with random router
4. **Train** Token+Global router (~1-2 hours)
5. Evaluate Token+Global with trained router
6. Compare all results

## Running Individual Methods

```bash
# 1. Linear Baseline
cd linear_baseline && bash run_merge.sh
python ../evaluate.py --model_path ./merged_linear --method_name linear

# 2. MoDEM (Model-Level Router)
cd model_router && python evaluate_model_router.py --compute_centroids

# 3. Token + Global (Untrained)
cd token_global_router
python evaluate_token_global.py --context_method momentum --fusion_method gate

# 4. Train Token + Global Router
cd token_global_router && bash run_training.sh

# 5. Token + Global (Trained)
cd token_global_router
python evaluate_token_global.py --checkpoint_path ./trained_router/final
```

## Folder Structure

```
moe_routing_experiment/
├── linear_baseline/           # Method 1: Static merge
│   ├── merge_config.yaml
│   └── run_merge.sh
├── model_router/              # Method 2: MoDEM-style
│   ├── model_level_router.py
│   └── evaluate_model_router.py
├── token_global_router/       # Method 3 & 4: Token + Global
│   ├── token_global_moe.py    # Model architecture
│   ├── train_router.py        # Training script
│   ├── evaluate_token_global.py
│   └── trained_router/        # Saved checkpoints
├── evaluate.py                # Unified evaluation
├── run_all_experiments.sh     # Run everything
├── results/                   # All results (JSON)
└── README.md
```

## Expected Output

After running all experiments:

```
========================================
FINAL COMPARISON
========================================
Method                          gsm8k    mmlu     humaneval
------------------------------------------------------------
linear                          0.XXXX   0.XXXX   0.XXXX
model_router                    0.XXXX   0.XXXX   0.XXXX
token_global_momentum_gate      0.XXXX   0.XXXX   0.XXXX
token_global_momentum_gate_trained  0.XXXX   0.XXXX   0.XXXX
```

## Key Hypothesis

If Token+Global (Trained) > MoDEM > Linear, it validates that:
1. Dynamic routing helps (MoDEM > Linear)
2. Token-level granularity with global context helps more (Token+Global > MoDEM)

## Citation

Related work:
- THOR-MoE: Token-level routing with context (NMT focused)
- MoDEM: Model-level domain expert mixture
