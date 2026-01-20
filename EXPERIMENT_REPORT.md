# MoE Routing Strategies Experiment Report

## Executive Summary

This experiment compared different Mixture-of-Experts (MoE) routing strategies for combining fine-tuned small language models (Qwen3-0.6B variants). **None of the sophisticated routing methods outperformed simple linear merging on aggregate benchmarks.** This negative result provides valuable insights for the research community.

---

## Experiment Setup
### Expert Models
- **Base**: Qwen/Qwen3-0.6B
- **Medical**: suayptalha/Qwen3-0.6B-Medical-Expert
- **Code**: suayptalha/Qwen3-0.6B-Code-Expert
- **Math**: suayptalha/Qwen3-0.6B-Math-Expert
- **Instruct**: suayptalha/Qwen3-0.6B-IF-Expert

### Methods Compared

| Method | Description | Routing Granularity |
|--------|-------------|---------------------|
| **Linear** | Static weight merge via mergekit | None (fixed weights) |
| **MoDEM** | Model-level routing via embedding similarity | Per-input |
| **Token+Global (Untrained)** | Token-level + attention-pooled context | Per-token |
| **Token+Global (Trained)** | Above with 3 epochs router training | Per-token |
| **Embedding Momentum** | Token-level + embedding model similarity bias | Per-token |

### Benchmarks
- **GSM8K**: Grade school math (100 samples)
- **MMLU**: Multi-task language understanding (100 samples)
- **HumanEval**: Code generation validity (50 samples)

---

## Results

| Method | GSM8K ↑ | MMLU ↑ | HumanEval ↑ |
|--------|---------|--------|-------------|
| Linear (mergekit) | 0.04 | **0.50** | 0.96 |
| MoDEM (model-level) | **0.15** | 0.39 | 0.96 |
| Token+Global (untrained) | 0.04 | 0.41 | **0.98** |
| Token+Global (trained) | 0.05 | 0.44 | 0.96 |
| Embedding Momentum (trained) | 0.07 | 0.42 | **0.98** |

### Observations

1. **Linear merge wins on MMLU** (0.50 vs 0.39-0.44 for routing methods)
2. **MoDEM wins on GSM8K** (0.15 vs 0.04-0.07) but still very low
3. **HumanEval is saturated** - all methods achieve 0.96-0.98
4. **Training provides minimal improvement** (+1-3% on MMLU, negligible elsewhere)
5. **More sophisticated routing ≠ better performance**

---

## Analysis: Why Routing Didn't Help

### 1. Expert Model Quality Issue

The fundamental problem: **the expert models themselves may not be specialized enough**.

- Math still only achieves ~15% on GSM8K via MoDEM
- This suggests the "experts" don't have strong domain-specific capabilities to route TO
- Merging may actually preserve more general capability than routing to weak specialists

### 2. Scale Matters

At 0.5-1.5B parameters:
- Models lack sufficient capacity for strong specialization
- Fine-tuning on domain data may cause catastrophic forgetting of general abilities
- The MoE overhead (multiple experts, routers) may not pay off at small scale

### 3. MoE Layer Selection

We applied MoE to middle layers (25%-75% of depth). This may be suboptimal:
- Early layers: feature extraction (shouldn't vary by domain)
- Middle layers: where we applied MoE
- Late layers: task-specific processing (might benefit more from routing)

### 4. Router Training Limitations

3 epochs of router training showed minimal improvement because:
- Expert MLPs are frozen - router can only learn to SELECT, not IMPROVE experts
- If experts are weak, better routing to weak experts won't help
- Load balancing loss may prevent router from specializing

### 5. Benchmark Mismatch

HumanEval near-saturation (0.96-0.98) suggests:
- Task too easy for these models (validity check, not correctness)
- Or all methods equally (in)capable at code generation
- Need harder benchmarks (HumanEval+ with test cases)

---

## What We Learned (Negative Results) so far

### ❌ Token-Level Routing Doesn't Beat Model-Level

Contrary to hypothesis, finer routing granularity didn't help:
- Token-level routing: 0.04-0.07 GSM8K
- Model-level routing: 0.15 GSM8K (best)

**Insight**: For small models, the overhead of per-token routing may not be worth it. Input-level routing (MoDEM-style) is simpler and performed better on math.

### ❌ Global Context Doesn't Improve Routing

Neither attention-pooled context nor embedding-model context improved performance:
- The "global semantic signal" may be redundant with what token-level routing already captures
- Or the context quality isn't good enough to provide useful routing signal

### ❌ Training Router ≠ Better Results

Router training provided minimal gains (~3% MMLU improvement):
- Frozen experts limit what training can achieve
- Need end-to-end training or better expert initialization

### ❌ Static Merging is a Strong Baseline

Linear merge achieved best MMLU (0.50) despite being the simplest method:
- Preserves knowledge from all experts uniformly
- No routing overhead or errors
- **For general-purpose use, simple merging may be optimal**

---

## Recommendations for Larger Models

### 1. Ensure Expert Quality First

Before implementing MoE routing:
- Verify each expert achieves strong performance on its domain independently
- If base_expert achieves X% and domain_expert achieves only X+5%, routing won't help much
- Target >20% absolute improvement from domain fine-tuning

### 2. Consider Model-Level Routing First

For practical applications:
- MoDEM-style model-level routing is simpler and showed best math performance
- Avoid token-level MoE complexity unless you have clear evidence it helps
- Model-level routing has O(1) overhead vs O(layers × tokens) for token-level

### 3. End-to-End Training is Critical

If using token-level MoE:
- Train experts AND routers together, not just routers
- Load balancing alone doesn't ensure expert specialization
- Consider auxiliary losses for expert specialization

### 4. Benchmark Selection Matters

For MoE evaluation:
- Use benchmarks where domain experts clearly outperform base
- Avoid saturated benchmarks (like HumanEval validity)
- Include domain-specific benchmarks that match your experts

### 5. Scale Considerations

| Model Size | Recommendation |
|------------|----------------|
| <1B | Static merge or model-level routing |
| 1-7B | Model-level routing, maybe sparse MoE |
| 7B+ | Token-level MoE may become beneficial |

---

## Comparison with Related Work

### vs. THOR-MoE (arXiv 2505.14173)
THOR uses hidden state averaging for global context. Our results suggest:
- The context computation method (attention pooling vs averaging vs embedding model) may not matter much
- The fundamental issue is expert quality, not routing mechanism

### vs. MoDEM
MoDEM's model-level routing actually performed best on GSM8K in our experiments, suggesting:
- Simpler approaches may be preferable for small models
- Model-level routing avoids compounding errors across layers

---

## Experimental Limitations

1. **Small sample sizes**: 100 samples per benchmark
2. **Limited expert variety**: Only 5 expert models
3. **Single model family**: All Qwen-based models
4. **Limited training**: Only 3 epochs, ~2000 samples
5. **No hyperparameter search**: Fixed momentum_weight=0.5, top_k=2

---

## Conclusion

**The sophisticated routing mechanisms we developed did not outperform simple linear merging.** This is a valuable negative result that suggests:

1. **Expert quality > routing sophistication** - Focus on building strong domain experts first
2. **Simple baselines are strong** - Linear merging should always be compared against
3. **Scale matters** - Token-level MoE may only pay off at larger scales
4. **Model-level routing is underrated** - MoDEM-style routing is simple and effective

For practitioners: **Start with linear merge. Only add routing complexity if you have verified strong domain experts and clear evidence that routing helps on your specific use case.**

---

## Files and Reproducibility

```
moe_routing_experiment/
├── linear_baseline/          # mergekit configuration
├── model_router/             # MoDEM implementation
├── token_global_router/      # Token+Global & Embedding Momentum
│   ├── token_global_moe.py
│   ├── embedding_momentum_moe.py
│   ├── train_router.py
│   └── train_embedding_momentum.py
├── evaluate.py               # Unified evaluation
├── run_all_experiments.sh    # Full experiment runner
└── results/                  # JSON result files
```

Run experiments:
```bash
bash run_all_experiments.sh
```

---

*Report generated: January 2026*
*Hardware: Kaggle T4 × 2*
*Total experiment time: ~4 hours*
