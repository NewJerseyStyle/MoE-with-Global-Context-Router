"""
Evaluation script for Embedding Momentum MoE

This evaluates the new momentum approach:
- Embedding model computes input semantics
- Cosine similarity to expert profiles = routing bias
- Token-level router + global bias = final routing
"""

import torch
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_momentum_moe import (
    build_embedding_momentum_moe,
    EXPERT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
)
from evaluate import evaluate_gsm8k, evaluate_mmlu, evaluate_humaneval


class EmbeddingMomentumWrapper:
    """Wrapper to make EmbeddingMomentumMoE compatible with evaluation functions."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.config = type('Config', (), {'pad_token_id': tokenizer.pad_token_id})()

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Set pad_token_id if not provided
        if 'pad_token_id' not in kwargs:
            kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        return self.model.generate(input_ids, attention_mask, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Embedding Momentum MoE")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--momentum_weight", type=float, default=0.5,
                       help="Weight for momentum bias (0=no momentum, 1=full momentum)")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["gsm8k", "mmlu", "humaneval"])
    parser.add_argument("--experts", type=str, nargs="+", default=None,
                       help="Which experts to use (default: all)")

    args = parser.parse_args()

    method_name = f"embedding_momentum_w{args.momentum_weight}"

    print("="*60)
    print(f"Embedding Momentum MoE Evaluation")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Momentum weight: {args.momentum_weight}")
    print(f"  Top-k: {args.top_k}")
    print("="*60)

    # Select experts
    if args.experts:
        expert_models = {k: v for k, v in EXPERT_MODELS.items() if k in args.experts}
    else:
        expert_models = EXPERT_MODELS

    print(f"\nUsing experts: {list(expert_models.keys())}")

    # Build model
    model, tokenizer = build_embedding_momentum_moe(
        expert_models=expert_models,
        embedding_model=args.embedding_model,
        momentum_weight=args.momentum_weight,
        top_k=args.top_k,
    )

    # Wrap for evaluation
    wrapped_model = EmbeddingMomentumWrapper(model, tokenizer)

    # Run evaluations
    results = {
        "method_name": method_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": args.embedding_model,
            "momentum_weight": args.momentum_weight,
            "top_k": args.top_k,
            "experts": list(expert_models.keys()),
        },
        "num_samples": args.num_samples,
        "benchmarks": {},
    }

    if "gsm8k" in args.benchmarks:
        print("\n" + "="*60)
        print("Evaluating GSM8K")
        print("="*60)
        results["benchmarks"]["gsm8k"] = evaluate_gsm8k(
            wrapped_model, tokenizer, num_samples=args.num_samples
        )

    if "mmlu" in args.benchmarks:
        print("\n" + "="*60)
        print("Evaluating MMLU")
        print("="*60)
        results["benchmarks"]["mmlu"] = evaluate_mmlu(
            wrapped_model, tokenizer, num_samples_per_subject=args.num_samples // 5
        )

    if "humaneval" in args.benchmarks:
        print("\n" + "="*60)
        print("Evaluating HumanEval")
        print("="*60)
        results["benchmarks"]["humaneval"] = evaluate_humaneval(
            wrapped_model, tokenizer, num_samples=min(args.num_samples, 50)
        )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for bench, res in results["benchmarks"].items():
        if "accuracy" in res:
            print(f"  {bench}: {res['accuracy']:.4f}")
        elif "validity_rate" in res:
            print(f"  {bench}: {res['validity_rate']:.4f}")

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
