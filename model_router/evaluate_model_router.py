"""
Evaluation script for Model-Level Router

This wraps the model-level router to work with the unified evaluation framework.
"""

import torch
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_level_router import ModelLevelMoE, create_default_domain_samples
from evaluate import evaluate_gsm8k, evaluate_mmlu, evaluate_humaneval
import json
import argparse
from datetime import datetime


class ModelRouterWrapper:
    """
    Wrapper to make ModelLevelMoE compatible with evaluation functions.
    Mimics the interface of a HuggingFace model.
    """
    def __init__(self, moe: ModelLevelMoE, default_expert: str = None):
        self.moe = moe
        self.default_expert = default_expert
        self.device = moe.device
        self.config = type('Config', (), {'pad_token_id': moe.tokenizer.pad_token_id})()

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate using routed expert."""
        # Route to select expert
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        selected_domains, weights = self.moe.route(input_ids, attention_mask, top_k=1)
        expert = selected_domains[0] if not self.default_expert else self.default_expert

        # Generate with selected expert
        return self.moe.generate_single(input_ids, attention_mask, expert, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model-Level Router")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--compute_centroids", action="store_true",
                       help="Compute domain centroids before evaluation")
    parser.add_argument("--centroids_path", type=str, default="domain_centroids.pt")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["gsm8k", "mmlu", "humaneval"])

    args = parser.parse_args()

    print("="*60)
    print("Model-Level Router Evaluation")
    print("="*60)

    # Initialize MoE
    moe = ModelLevelMoE(router_type="similarity", load_in_4bit=True)
    moe.load_base_for_routing()

    # Setup centroids
    centroids_file = Path(args.centroids_path)
    if args.compute_centroids or not centroids_file.exists():
        print("\nComputing domain centroids...")
        samples = create_default_domain_samples()
        moe.compute_domain_centroids(samples, save_path=str(centroids_file))
    else:
        print(f"\nLoading centroids from {centroids_file}")
        moe.load_domain_centroids(str(centroids_file))

    # Create wrapper for evaluation
    model = ModelRouterWrapper(moe)
    tokenizer = moe.tokenizer

    # Run evaluations
    results = {
        "method_name": "model_router",
        "timestamp": datetime.now().isoformat(),
        "num_samples": args.num_samples,
        "benchmarks": {},
        "routing_stats": {}
    }

    if "gsm8k" in args.benchmarks:
        print("\n" + "="*60)
        print("Evaluating GSM8K (will route to math expert)")
        print("="*60)
        results["benchmarks"]["gsm8k"] = evaluate_gsm8k(
            model, tokenizer, num_samples=args.num_samples
        )

    if "mmlu" in args.benchmarks:
        print("\n" + "="*60)
        print("Evaluating MMLU (will route based on subject)")
        print("="*60)
        results["benchmarks"]["mmlu"] = evaluate_mmlu(
            model, tokenizer, num_samples_per_subject=args.num_samples // 5
        )

    if "humaneval" in args.benchmarks:
        print("\n" + "="*60)
        print("Evaluating HumanEval (will route to code expert)")
        print("="*60)
        results["benchmarks"]["humaneval"] = evaluate_humaneval(
            model, tokenizer, num_samples=min(args.num_samples, 50)
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

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"model_router_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
