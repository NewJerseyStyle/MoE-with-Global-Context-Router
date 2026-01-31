"""
Evaluation script for Token + Global Context MoE

Evaluates our main contribution: token-level routing with sequence-level context.
Supports loading trained router checkpoints.
"""

import torch
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from token_global_moe import build_token_global_moe, EXPERT_MODELS
from evaluate import evaluate_gsm8k, evaluate_mmlu, evaluate_humaneval
from train_router import load_checkpoint


class TokenGlobalWrapper:
    """Wrapper to make TokenGlobalMoE compatible with evaluation functions."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.config = type('Config', (), {'pad_token_id': tokenizer.pad_token_id})()

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(input_ids, attention_mask, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Token + Global Context MoE")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--context_method", type=str, default="momentum",
                       choices=["momentum", "thor", "combined"])
    parser.add_argument("--fusion_method", type=str, default="gate",
                       choices=["gate", "add", "concat"])
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["gsm8k", "mmlu", "humaneval"])
    parser.add_argument("--experts", type=str, nargs="+", default=None,
                       help="Which experts to use (default: all)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to trained router checkpoint")

    args = parser.parse_args()

    trained = args.checkpoint_path is not None
    method_name = f"token_global_{args.context_method}_{args.fusion_method}"
    if trained:
        method_name += "_trained"

    print("="*60)
    print(f"Token + Global Context MoE Evaluation")
    print(f"  Context: {args.context_method}")
    print(f"  Fusion: {args.fusion_method}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Trained: {trained}")
    print("="*60)

    # Select experts
    if args.experts:
        expert_models = {k: v for k, v in EXPERT_MODELS.items() if k in args.experts}
    else:
        expert_models = EXPERT_MODELS

    print(f"\nUsing experts: {list(expert_models.keys())}")

    # Build model
    model, tokenizer = build_token_global_moe(
        expert_models=expert_models,
        context_method=args.context_method,
        fusion_method=args.fusion_method,
        context_dim=args.context_dim,
        top_k=args.top_k,
    )

    # Load trained checkpoint if provided
    if args.checkpoint_path:
        print(f"\nLoading trained checkpoint from: {args.checkpoint_path}")
        model = load_checkpoint(model, Path(args.checkpoint_path))

    # Wrap for evaluation
    wrapped_model = TokenGlobalWrapper(model, tokenizer)

    # Run evaluations
    results = {
        "method_name": method_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_method": args.context_method,
            "fusion_method": args.fusion_method,
            "top_k": args.top_k,
            "context_dim": args.context_dim,
            "experts": list(expert_models.keys()),
            "trained": trained,
            "checkpoint_path": args.checkpoint_path,
        },
        "num_samples": args.num_samples,
        "benchmarks": {},
    }

    if "gsm8k" in args.benchmarks:
        results["benchmarks"]["gsm8k"] = evaluate_gsm8k(
            wrapped_model, tokenizer, num_samples=args.num_samples
        )

    if "mmlu" in args.benchmarks:
        results["benchmarks"]["mmlu"] = evaluate_mmlu(
            wrapped_model, tokenizer, num_samples_per_subject=args.num_samples // 5
        )

    if "humaneval" in args.benchmarks:
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
