"""
Unified Benchmark Evaluation Script for MoE Routing Experiments

Evaluates models on multiple benchmarks and saves results for comparison.

Usage:
    python evaluate.py --model_path ./linear_baseline/merged_linear --method_name linear
    python evaluate.py --model_path Qwen/Qwen3-0.6B --method_name base --num_samples 50
"""

import torch
import json
import re
import argparse
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


def load_model(model_path: str, use_flash_attn: bool = False) -> Tuple:
    """Load model and tokenizer with optimal settings."""
    print(f"Loading model: {model_path}")

    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "attn_implementation": "eager",  # Explicitly disable flash attention
    }

    # Try flash attention if requested and GPU supports it (Ampere+)
    if use_flash_attn:
        try:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 8:  # Ampere (SM80) or newer
                    kwargs["attn_implementation"] = "flash_attention_2"
                    print("Using Flash Attention 2")
                else:
                    print(f"GPU compute capability {capability[0]}.{capability[1]} < 8.0, using eager attention")
        except Exception as e:
            print(f"Flash Attention check failed: {e}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def evaluate_gsm8k(model, tokenizer, num_samples: int = 100, num_shots: int = 5) -> Dict:
    """Evaluate on GSM8K math benchmark."""
    print(f"\n{'='*60}")
    print(f"GSM8K Evaluation (n={num_samples}, {num_shots}-shot)")
    print(f"{'='*60}")

    dataset = load_dataset("gsm8k", "main", split="test")
    train_data = load_dataset("gsm8k", "main", split="train")

    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    # Build few-shot prompt
    few_shot_examples = []
    for i in range(min(num_shots, len(train_data))):
        q = train_data[i]["question"]
        a = train_data[i]["answer"]
        few_shot_examples.append(f"Question: {q}\nAnswer: {a}")
    few_shot_prompt = "\n\n".join(few_shot_examples) + "\n\n"

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="GSM8K"):
        question = item["question"]
        answer = item["answer"]
        true_answer = answer.split("####")[-1].strip()

        prompt = few_shot_prompt + f"Question: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract answer
        if "####" in generated:
            pred_answer = generated.split("####")[-1].strip().split()[0] if generated.split("####")[-1].strip() else ""
        else:
            numbers = re.findall(r'-?\d+\.?\d*', generated)
            pred_answer = numbers[-1] if numbers else ""

        # Compare
        try:
            true_val = float(true_answer.replace(",", ""))
            pred_val = float(pred_answer.replace(",", "")) if pred_answer else float('nan')
            if abs(true_val - pred_val) < 0.01:
                correct += 1
        except:
            if true_answer.strip() == pred_answer.strip():
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"GSM8K Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_mmlu(model, tokenizer, num_samples_per_subject: int = 20, num_shots: int = 5) -> Dict:
    """Evaluate on MMLU knowledge benchmark."""
    print(f"\n{'='*60}")
    print(f"MMLU Evaluation ({num_shots}-shot)")
    print(f"{'='*60}")

    # Representative subjects across domains
    subjects = [
        "abstract_algebra",      # Math
        "college_physics",       # Science
        "computer_security",     # CS
        "clinical_knowledge",    # Medical
        "high_school_mathematics",
    ]

    all_correct = 0
    all_total = 0
    subject_results = {}

    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            val_data = load_dataset("cais/mmlu", subject, split="validation")
        except Exception as e:
            print(f"  Skipping {subject}: {e}")
            continue

        if num_samples_per_subject < len(dataset):
            dataset = dataset.select(range(num_samples_per_subject))

        # Few-shot prompt
        few_shot = ""
        for i in range(min(num_shots, len(val_data))):
            item = val_data[i]
            q = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            answer_letter = ["A", "B", "C", "D"][answer_idx]
            choice_text = "\n".join([f"{['A','B','C','D'][j]}. {c}" for j, c in enumerate(choices)])
            few_shot += f"Question: {q}\n{choice_text}\nAnswer: {answer_letter}\n\n"

        correct = 0
        total = 0

        for item in dataset:
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            true_answer = ["A", "B", "C", "D"][answer_idx]

            choice_text = "\n".join([f"{['A','B','C','D'][j]}. {c}" for j, c in enumerate(choices)])
            prompt = few_shot + f"Question: {question}\n{choice_text}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_answer = generated.strip()[0].upper() if generated.strip() else ""

            if pred_answer == true_answer:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        subject_results[subject] = {"accuracy": accuracy, "correct": correct, "total": total}
        all_correct += correct
        all_total += total

        print(f"  {subject}: {accuracy:.4f} ({correct}/{total})")

    overall_accuracy = all_correct / all_total if all_total > 0 else 0
    print(f"MMLU Overall: {overall_accuracy:.4f} ({all_correct}/{all_total})")

    return {
        "accuracy": overall_accuracy,
        "correct": all_correct,
        "total": all_total,
        "subject_results": subject_results
    }


def evaluate_humaneval(model, tokenizer, num_samples: int = 30) -> Dict:
    """Evaluate on HumanEval code generation (simplified without execution)."""
    print(f"\n{'='*60}")
    print(f"HumanEval Evaluation (n={num_samples}, no execution)")
    print(f"{'='*60}")

    dataset = load_dataset("openai_humaneval", split="test")

    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    results = []

    for item in tqdm(dataset, desc="HumanEval"):
        prompt = item["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Simple validity heuristics
        has_return = "return" in generated
        no_redefine = "def " not in generated[:20]
        has_content = len(generated.strip()) > 10
        looks_valid = has_return and no_redefine and has_content

        results.append({
            "task_id": item["task_id"],
            "looks_valid": looks_valid
        })

    valid_count = sum(1 for r in results if r["looks_valid"])
    validity_rate = valid_count / len(results) if results else 0

    print(f"HumanEval Validity Rate: {validity_rate:.4f} ({valid_count}/{len(results)})")
    print("(Note: This is NOT pass@k - no code execution)")

    return {
        "validity_rate": validity_rate,
        "valid_count": valid_count,
        "total": len(results)
    }


def evaluate_medical(model, tokenizer, num_samples: int = 50) -> Dict:
    """Evaluate on medical QA (MedQA subset)."""
    print(f"\n{'='*60}")
    print(f"Medical QA Evaluation (n={num_samples})")
    print(f"{'='*60}")

    try:
        # Try to load MedQA or similar medical dataset
        dataset = load_dataset("bigbio/med_qa", "med_qa_en_source", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Could not load MedQA: {e}")
        print("Skipping medical evaluation")
        return {"accuracy": None, "error": str(e)}

    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="MedQA"):
        question = item.get("question", item.get("sent1", ""))
        options = item.get("options", item.get("choices", {}))
        answer = item.get("answer_idx", item.get("answer", ""))

        if isinstance(options, dict):
            choice_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        elif isinstance(options, list):
            choice_text = "\n".join([f"{['A','B','C','D','E'][i]}. {c}" for i, c in enumerate(options)])
        else:
            continue

        prompt = f"Medical Question: {question}\n{choice_text}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = generated.strip()[0].upper() if generated.strip() else ""

        true_answer = str(answer).upper() if answer else ""

        if pred_answer == true_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Medical QA Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def run_evaluation(
    model_path: str,
    method_name: str,
    output_dir: str = "./results",
    num_samples: int = 100,
    benchmarks: List[str] = None
) -> Dict:
    """Run full evaluation suite and save results."""

    if benchmarks is None:
        benchmarks = ["gsm8k", "mmlu", "humaneval"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_path)

    # Run evaluations
    results = {
        "method_name": method_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "num_samples": num_samples,
        "benchmarks": {}
    }

    start_time = time.time()

    if "gsm8k" in benchmarks:
        results["benchmarks"]["gsm8k"] = evaluate_gsm8k(model, tokenizer, num_samples=num_samples)

    if "mmlu" in benchmarks:
        results["benchmarks"]["mmlu"] = evaluate_mmlu(model, tokenizer, num_samples_per_subject=num_samples // 5)

    if "humaneval" in benchmarks:
        results["benchmarks"]["humaneval"] = evaluate_humaneval(model, tokenizer, num_samples=min(num_samples, 50))

    if "medical" in benchmarks:
        results["benchmarks"]["medical"] = evaluate_medical(model, tokenizer, num_samples=num_samples)

    results["total_time_seconds"] = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {method_name}")
    print(f"{'='*60}")

    for bench, res in results["benchmarks"].items():
        if "accuracy" in res and res["accuracy"] is not None:
            print(f"  {bench}: {res['accuracy']:.4f}")
        elif "validity_rate" in res:
            print(f"  {bench}: {res['validity_rate']:.4f} (validity)")

    print(f"\nTotal time: {results['total_time_seconds']:.1f}s")

    # Save results
    result_file = output_path / f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {result_file}")

    return results


def compare_results(results_dir: str = "./results"):
    """Load and compare all evaluation results."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print("No results directory found")
        return

    all_results = []
    for f in results_path.glob("*.json"):
        with open(f) as fp:
            all_results.append(json.load(fp))

    if not all_results:
        print("No results found")
        return

    # Group by method and get latest
    latest_by_method = {}
    for r in all_results:
        method = r["method_name"]
        if method not in latest_by_method or r["timestamp"] > latest_by_method[method]["timestamp"]:
            latest_by_method[method] = r

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    methods = list(latest_by_method.keys())
    benchmarks = set()
    for r in latest_by_method.values():
        benchmarks.update(r["benchmarks"].keys())
    benchmarks = sorted(benchmarks)

    # Header
    header = f"{'Method':<20}"
    for b in benchmarks:
        header += f"{b:<15}"
    print(header)
    print("-" * 80)

    # Rows
    for method in sorted(methods):
        r = latest_by_method[method]
        row = f"{method:<20}"
        for b in benchmarks:
            if b in r["benchmarks"]:
                res = r["benchmarks"][b]
                if "accuracy" in res and res["accuracy"] is not None:
                    row += f"{res['accuracy']:.4f}         "
                elif "validity_rate" in res:
                    row += f"{res['validity_rate']:.4f}         "
                else:
                    row += f"{'N/A':<15}"
            else:
                row += f"{'--':<15}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--method_name", type=str, required=True, help="Name for this method (e.g., 'linear', 'momentum')")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=100, help="Samples per benchmark")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["gsm8k", "mmlu", "humaneval"],
                       choices=["gsm8k", "mmlu", "humaneval", "medical"],
                       help="Which benchmarks to run")
    parser.add_argument("--compare", action="store_true", help="Compare existing results instead of running evaluation")

    args = parser.parse_args()

    if args.compare:
        compare_results(args.output_dir)
    else:
        run_evaluation(
            model_path=args.model_path,
            method_name=args.method_name,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            benchmarks=args.benchmarks
        )


if __name__ == "__main__":
    main()
