#!/usr/bin/env python3
"""
Evaluation Script for Reversal Curse Research

This script evaluates trained models on the reversal curse phenomenon:
1. Tests forward direction knowledge (A -> B)
2. Tests reverse direction knowledge (B -> A)
3. Compares performance to measure reversal curse severity

Usage:
    python evaluate_reversal.py --model-path ./results/baseline --eval-file ./data/eval/reversal_curse_eval.jsonl
    python evaluate_reversal.py compare --baseline ./results/baseline --experimental ./results/experimental
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch


def load_model_and_tokenizer(model_path: str, base_model: Optional[str] = None):
    """
    Load a trained model and tokenizer.

    Args:
        model_path: Path to the trained model (LoRA adapter)
        base_model: Base model name (if not specified, tries to infer)

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Try to load config to get base model
    config_path = Path(model_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            base_model = base_model or config.get("base_model_name_or_path")

    if not base_model:
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Using default base model: {base_model}")

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    return response


def check_answer_correctness(response: str, expected_entities: list[str]) -> dict:
    """
    Check if the response contains expected entities.

    Returns:
        Dictionary with correctness metrics
    """
    response_lower = response.lower()
    found_entities = []
    missing_entities = []

    for entity in expected_entities:
        if entity.lower() in response_lower:
            found_entities.append(entity)
        else:
            missing_entities.append(entity)

    return {
        "correct": len(missing_entities) == 0,
        "found_entities": found_entities,
        "missing_entities": missing_entities,
        "score": len(found_entities) / len(expected_entities) if expected_entities else 1.0
    }


def evaluate_reversal_curse(model, tokenizer, eval_data: list[dict]) -> dict:
    """
    Evaluate the model on reversal curse test cases.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        eval_data: List of evaluation examples

    Returns:
        Dictionary with evaluation results
    """
    results = {
        "forward": {"correct": 0, "total": 0, "scores": [], "examples": []},
        "reverse": {"correct": 0, "total": 0, "scores": [], "examples": []},
    }

    print("\n" + "=" * 60)
    print("REVERSAL CURSE EVALUATION")
    print("=" * 60)

    for i, example in enumerate(eval_data):
        print(f"\n--- Example {i + 1} ---")
        print(f"Relation: {example.get('relation', 'unknown')}")

        # Test forward direction
        forward_q = example["forward_q"]
        print(f"\nForward Q: {forward_q}")
        forward_response = generate_response(model, tokenizer, forward_q)
        print(f"Response: {forward_response}")

        forward_check = check_answer_correctness(
            forward_response,
            [example.get("entity_a", ""), example.get("entity_b", "")]
        )
        print(f"Correct: {forward_check['correct']} (score: {forward_check['score']:.2f})")

        results["forward"]["total"] += 1
        results["forward"]["scores"].append(forward_check["score"])
        if forward_check["correct"]:
            results["forward"]["correct"] += 1
        results["forward"]["examples"].append({
            "question": forward_q,
            "response": forward_response,
            "check": forward_check
        })

        # Test reverse direction
        reverse_q = example["reverse_q"]
        print(f"\nReverse Q: {reverse_q}")
        reverse_response = generate_response(model, tokenizer, reverse_q)
        print(f"Response: {reverse_response}")

        reverse_check = check_answer_correctness(
            reverse_response,
            [example.get("entity_a", ""), example.get("entity_b", "")]
        )
        print(f"Correct: {reverse_check['correct']} (score: {reverse_check['score']:.2f})")

        results["reverse"]["total"] += 1
        results["reverse"]["scores"].append(reverse_check["score"])
        if reverse_check["correct"]:
            results["reverse"]["correct"] += 1
        results["reverse"]["examples"].append({
            "question": reverse_q,
            "response": reverse_response,
            "check": reverse_check
        })

    # Calculate summary statistics
    results["forward"]["accuracy"] = results["forward"]["correct"] / results["forward"]["total"] if results["forward"]["total"] > 0 else 0
    results["reverse"]["accuracy"] = results["reverse"]["correct"] / results["reverse"]["total"] if results["reverse"]["total"] > 0 else 0
    results["forward"]["avg_score"] = sum(results["forward"]["scores"]) / len(results["forward"]["scores"]) if results["forward"]["scores"] else 0
    results["reverse"]["avg_score"] = sum(results["reverse"]["scores"]) / len(results["reverse"]["scores"]) if results["reverse"]["scores"] else 0

    # Reversal curse severity
    results["reversal_curse_gap"] = results["forward"]["accuracy"] - results["reverse"]["accuracy"]

    return results


def print_summary(results: dict, model_name: str = "Model"):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("=" * 60)
    print(f"\nForward Direction (A -> B):")
    print(f"  Accuracy: {results['forward']['accuracy']:.1%} ({results['forward']['correct']}/{results['forward']['total']})")
    print(f"  Avg Score: {results['forward']['avg_score']:.2f}")

    print(f"\nReverse Direction (B -> A):")
    print(f"  Accuracy: {results['reverse']['accuracy']:.1%} ({results['reverse']['correct']}/{results['reverse']['total']})")
    print(f"  Avg Score: {results['reverse']['avg_score']:.2f}")

    print(f"\nReversal Curse Gap: {results['reversal_curse_gap']:.1%}")
    print("  (Positive = forward better than reverse = reversal curse present)")
    print("=" * 60)


def compare_models(baseline_results: dict, experimental_results: dict):
    """Compare baseline and experimental model results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("\n                     Baseline    Experimental    Improvement")
    print("-" * 60)

    # Forward accuracy
    b_fwd = baseline_results['forward']['accuracy']
    e_fwd = experimental_results['forward']['accuracy']
    print(f"Forward Accuracy:    {b_fwd:.1%}       {e_fwd:.1%}           {e_fwd - b_fwd:+.1%}")

    # Reverse accuracy
    b_rev = baseline_results['reverse']['accuracy']
    e_rev = experimental_results['reverse']['accuracy']
    print(f"Reverse Accuracy:    {b_rev:.1%}       {e_rev:.1%}           {e_rev - b_rev:+.1%}")

    # Reversal curse gap
    b_gap = baseline_results['reversal_curse_gap']
    e_gap = experimental_results['reversal_curse_gap']
    print(f"Reversal Curse Gap:  {b_gap:.1%}       {e_gap:.1%}           {b_gap - e_gap:+.1%} reduction")

    print("-" * 60)

    # Analysis
    print("\nANALYSIS:")
    if e_gap < b_gap:
        reduction = (b_gap - e_gap) / b_gap * 100 if b_gap != 0 else 0
        print(f"  - Bidirectional training REDUCED reversal curse by {reduction:.1f}%")
    elif e_gap > b_gap:
        increase = (e_gap - b_gap) / b_gap * 100 if b_gap != 0 else 0
        print(f"  - Bidirectional training INCREASED reversal curse by {increase:.1f}%")
    else:
        print("  - No significant change in reversal curse")

    if e_rev > b_rev:
        print(f"  - Reverse direction improved by {(e_rev - b_rev):.1%}")
    elif e_rev < b_rev:
        print(f"  - Reverse direction degraded by {(b_rev - e_rev):.1%}")

    print("=" * 60)


def load_eval_data(eval_file: str) -> list[dict]:
    """Load evaluation data from JSONL file."""
    data = []
    with open(eval_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate reversal curse in LLMs")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Evaluate single model
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a single model')
    eval_parser.add_argument('--model-path', required=True,
                            help='Path to trained model (LoRA adapter)')
    eval_parser.add_argument('--base-model', default=None,
                            help='Base model name')
    eval_parser.add_argument('--eval-file', required=True,
                            help='Path to evaluation JSONL file')
    eval_parser.add_argument('--output', default=None,
                            help='Output file for results (JSON)')

    # Compare two models
    compare_parser = subparsers.add_parser('compare', help='Compare baseline vs experimental')
    compare_parser.add_argument('--baseline', required=True,
                               help='Path to baseline model')
    compare_parser.add_argument('--experimental', required=True,
                               help='Path to experimental model')
    compare_parser.add_argument('--base-model', default=None,
                               help='Base model name')
    compare_parser.add_argument('--eval-file', required=True,
                               help='Path to evaluation JSONL file')
    compare_parser.add_argument('--output', default=None,
                               help='Output file for results (JSON)')

    # Create sample eval data
    create_parser = subparsers.add_parser('create-eval', help='Create sample evaluation data')
    create_parser.add_argument('--output', default='./data/eval/reversal_curse_eval.jsonl',
                              help='Output file path')

    args = parser.parse_args()

    if args.command == 'evaluate':
        print(f"Loading evaluation data from: {args.eval_file}")
        eval_data = load_eval_data(args.eval_file)
        print(f"Loaded {len(eval_data)} evaluation examples")

        print(f"\nLoading model from: {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

        results = evaluate_reversal_curse(model, tokenizer, eval_data)
        print_summary(results, args.model_path)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.command == 'compare':
        print(f"Loading evaluation data from: {args.eval_file}")
        eval_data = load_eval_data(args.eval_file)

        print(f"\n--- Evaluating Baseline Model ---")
        baseline_model, tokenizer = load_model_and_tokenizer(args.baseline, args.base_model)
        baseline_results = evaluate_reversal_curse(baseline_model, tokenizer, eval_data)
        print_summary(baseline_results, "Baseline (2x Forward)")

        # Free memory
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n--- Evaluating Experimental Model ---")
        exp_model, tokenizer = load_model_and_tokenizer(args.experimental, args.base_model)
        exp_results = evaluate_reversal_curse(exp_model, tokenizer, eval_data)
        print_summary(exp_results, "Experimental (Forward + Reversed)")

        compare_models(baseline_results, exp_results)

        if args.output:
            comparison = {
                "baseline": baseline_results,
                "experimental": exp_results,
            }
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.command == 'create-eval':
        from prepare_dataset import create_eval_dataset
        create_eval_dataset(os.path.dirname(args.output))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
