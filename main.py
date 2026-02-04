#!/usr/bin/env python3
"""
Tool Selection Attention Analysis - Main Entry Point

Usage:
    python main.py test          # Run setup verification tests
    python main.py baseline      # Check baseline accuracy
    python main.py phase1        # Run Phase 1 analysis
    python main.py phase2        # Run Phase 2 ablation (requires phase1 results)
    python main.py phase2 --n-heads 10  # Ablate top 10 heads
    python main.py phase2 --control     # Ablate bottom heads (control experiment)
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools import TOOLS, get_tool_names
from src.prompts import format_prompt_with_tools
from src.parsing import parse_tool_calls, get_first_tool_name
from src.tokens import find_tool_token_spans
from src.attention import AttentionExtractor
from src.dataset import load_curated_dataset, get_dataset_stats
from src.analysis import (
    run_experiment,
    compute_head_tool_correlations,
    rank_heads_by_correlation,
    compute_accuracy,
    save_results,
)
from src.ablation import (
    run_ablation_experiment,
    load_top_heads,
    load_bottom_heads,
)


MODELS = {
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
}
DEFAULT_MODEL = "8b"


def get_model_name(args) -> str:
    """Get model name from args, defaulting to DEFAULT_MODEL."""
    model_key = getattr(args, "model", DEFAULT_MODEL)
    return MODELS[model_key]


def cmd_test(args):
    """Run setup verification tests."""
    model_name = get_model_name(args)
    print("=" * 60)
    print("Tool Selection Attention Analysis - Setup Verification")
    print("=" * 60)
    print(f"Model: {model_name}")

    # Test 1: Prompt formatting
    print("\n[1/5] Testing prompt formatting...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "What's the weather in Tokyo?"
    formatted = format_prompt_with_tools(tokenizer, prompt, TOOLS)
    assert "get_weather" in formatted
    print(f"  Formatted prompt length: {len(formatted)} chars")
    print("  ✓ Prompt formatting works")

    # Test 2: Token mapping
    print("\n[2/5] Testing token mapping...")
    tool_spans = find_tool_token_spans(tokenizer, formatted, TOOLS)
    assert len(tool_spans) == 10
    assert tool_spans[0].name_span is not None
    print(f"  Found spans for {len(tool_spans)} tools")
    for ts in tool_spans[:3]:
        print(f"    {ts.tool_name}: {ts.total_tokens} tokens")
    print("  ✓ Token mapping works")

    # Test 3: Tool parsing
    print("\n[3/5] Testing tool call parsing...")
    sample = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'
    calls = parse_tool_calls(sample)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    print("  ✓ Tool parsing works")

    # Test 4: Dataset
    print("\n[4/5] Testing dataset...")
    dataset = load_curated_dataset()
    stats = get_dataset_stats(dataset)
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Tools: {stats['tools']}")
    print(f"  Prompts per tool: {stats['min_per_tool']}-{stats['max_per_tool']}")
    print("  ✓ Dataset works")

    # Test 5: GPU tests (optional)
    if args.gpu:
        print("\n[5/5] Testing GPU components...")

        print("  Loading model...")
        extractor = AttentionExtractor(model_name)
        extractor.load(use_nnsight=False)
        print(f"  Model: {extractor.n_layers} layers, {extractor.n_heads} heads")

        print("  Extracting attention...")
        attention = extractor.extract_attention(formatted)
        print(f"  Attention shape: {attention.shape}")
        print("  ✓ Attention extraction works")

        print("\n  Testing generation...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:])
        predicted = get_first_tool_name(response)
        print(f"  Predicted tool: {predicted}")
        if predicted == "get_weather":
            print("  ✓ Generation works correctly")
        else:
            print("  ⚠ Unexpected tool selected")
    else:
        print("\n[5/5] Skipping GPU tests (use --gpu to enable)")

    print("\n" + "=" * 60)
    print("Setup verification complete!")
    print("=" * 60)


def cmd_baseline(args):
    """Check baseline accuracy on the dataset."""
    model_name = get_model_name(args)
    print("=" * 60)
    print("Baseline Accuracy Check")
    print("=" * 60)

    dataset = load_curated_dataset()
    random.seed(42)
    random.shuffle(dataset)
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    correct = 0
    total = len(dataset)
    tool_results = {name: {"correct": 0, "total": 0} for name in get_tool_names()}
    failures = []

    print(f"\nRunning {total} prompts...")
    for i, eval_prompt in enumerate(dataset):
        formatted = format_prompt_with_tools(tokenizer, eval_prompt.prompt, TOOLS)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,  # High limit, stop_strings will cut it short
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["</tool_call>"],
                tokenizer=tokenizer,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:])
        predicted = get_first_tool_name(response)

        is_correct = predicted == eval_prompt.expected_tool
        if is_correct:
            correct += 1
        else:
            failures.append({
                "prompt": eval_prompt.prompt,
                "expected": eval_prompt.expected_tool,
                "predicted": predicted,
                "response_snippet": response[:200],
            })

        tool_results[eval_prompt.expected_tool]["total"] += 1
        if is_correct:
            tool_results[eval_prompt.expected_tool]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total} ({correct}/{i+1} correct)")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nOverall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    print("\nPer-tool accuracy:")
    for tool, stats in tool_results.items():
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            print(f"  {tool}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    if failures:
        print("\n" + "=" * 60)
        print(f"Failure Analysis ({len(failures)} failures)")
        print("=" * 60)
        for f in failures[:10]:  # Show first 10
            print(f"\nPrompt: {f['prompt']}")
            print(f"Expected: {f['expected']}")
            print(f"Predicted: {f['predicted']}")
            if f['predicted'] is None:
                print(f"Response: {f['response_snippet']}...")


def cmd_phase1(args):
    """Run Phase 1 analysis."""
    model_name = get_model_name(args)
    print("=" * 60)
    print("Phase 1: Find Candidate Heads")
    print("=" * 60)
    print(f"Model: {model_name}")

    # Load dataset
    dataset = load_curated_dataset()
    random.seed(42)
    random.shuffle(dataset)
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
        print(f"\nUsing {len(dataset)} prompts (limited, shuffled)")
    else:
        print(f"\nUsing full dataset: {len(dataset)} prompts")

    # Run experiment
    print("\nRunning experiment...")
    results = run_experiment(
        dataset=dataset,
        model_name=model_name,
        tools=TOOLS,
    )

    # Compute metrics
    print("\nComputing correlations...")
    accuracy = compute_accuracy(results)
    correlations = compute_head_tool_correlations(results, TOOLS)
    ranked = rank_heads_by_correlation(correlations)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\nOverall accuracy: {100*accuracy['overall_accuracy']:.1f}%")

    print("\nTop 20 Tool Selection Heads:")
    print("-" * 40)
    for _, row in ranked.head(20).iterrows():
        print(f"  {row['rank']:2d}. {row['head_id']:8s}  corr={row['correlation']:.4f}")

    # Save results
    output_dir = Path(args.output)
    save_results(results, correlations, ranked, accuracy, output_dir)

    print(f"\nResults saved to {output_dir}/")


def cmd_phase2(args):
    """Run Phase 2 ablation experiment."""
    model_name = get_model_name(args)
    print("=" * 60)
    print("Phase 2: Causal Validation (Ablation)")
    print("=" * 60)
    print(f"Model: {model_name}")

    # Load dataset
    dataset = load_curated_dataset()
    random.seed(42)
    random.shuffle(dataset)
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
        print(f"\nUsing {len(dataset)} prompts")
    else:
        print(f"\nUsing full dataset: {len(dataset)} prompts")

    # Load heads from Phase 1 results
    ranked_path = Path(args.phase1_results) / "ranked_heads.csv"
    if not ranked_path.exists():
        print(f"Error: Phase 1 results not found at {ranked_path}")
        print("Run 'python main.py phase1' first.")
        sys.exit(1)

    # Determine which heads to ablate
    if args.heads:
        # Parse custom head specification like "L20H15,L21H11"
        heads_to_ablate = []
        for h in args.heads.split(","):
            h = h.strip()
            if h.startswith("L") and "H" in h:
                layer = int(h.split("H")[0][1:])
                head = int(h.split("H")[1])
                heads_to_ablate.append((layer, head))
        print(f"\nAblating custom heads: {heads_to_ablate}")
    elif args.control:
        heads_to_ablate = load_bottom_heads(ranked_path, args.n_heads)
        print(f"\nControl: Ablating bottom {args.n_heads} heads")
    else:
        heads_to_ablate = load_top_heads(ranked_path, args.n_heads)
        print(f"\nAblating top {args.n_heads} heads")

    print(f"Heads: {heads_to_ablate}")

    # Load baseline from Phase 1 if skipping
    baseline_accuracy = None
    per_tool_baseline = None
    if args.skip_baseline:
        accuracy_path = Path(args.phase1_results) / "accuracy.json"
        if not accuracy_path.exists():
            print(f"Error: Phase 1 accuracy not found at {accuracy_path}")
            print("Cannot skip baseline without Phase 1 results.")
            sys.exit(1)
        import json
        with open(accuracy_path) as f:
            phase1_acc = json.load(f)
        baseline_accuracy = phase1_acc["overall_accuracy"]
        per_tool_baseline = phase1_acc["per_tool_accuracy"]
        print(f"Using Phase 1 baseline: {100*baseline_accuracy:.1f}%")

    # Run ablation experiment
    result = run_ablation_experiment(
        dataset=dataset,
        model_name=model_name,
        heads_to_ablate=heads_to_ablate,
        tools=TOOLS,
        baseline_accuracy=baseline_accuracy,
        per_tool_baseline=per_tool_baseline,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\nBaseline accuracy: {100*result.baseline_accuracy:.1f}%")
    print(f"Ablated accuracy:  {100*result.ablated_accuracy:.1f}%")
    print(f"Accuracy drop:     {100*result.accuracy_drop:.1f}%")

    print("\nPer-tool comparison:")
    print("-" * 50)
    print(f"{'Tool':<20} {'Baseline':>10} {'Ablated':>10} {'Drop':>10}")
    print("-" * 50)
    for tool in sorted(result.per_tool_baseline.keys()):
        base = result.per_tool_baseline[tool]
        abl = result.per_tool_ablated.get(tool, 0)
        drop = base - abl
        print(f"{tool:<20} {100*base:>9.1f}% {100*abl:>9.1f}% {100*drop:>+9.1f}%")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    results_dict = {
        "ablated_heads": [f"L{layer}H{head}" for layer, head in result.ablated_heads],
        "baseline_accuracy": result.baseline_accuracy,
        "ablated_accuracy": result.ablated_accuracy,
        "accuracy_drop": result.accuracy_drop,
        "per_tool_baseline": result.per_tool_baseline,
        "per_tool_ablated": result.per_tool_ablated,
    }
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Tool Selection Attention Analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common model argument
    model_choices = list(MODELS.keys())
    model_help = f"Model to use: {', '.join(f'{k}={v}' for k, v in MODELS.items())} (default: {DEFAULT_MODEL})"

    # Test command
    test_parser = subparsers.add_parser("test", help="Run setup verification")
    test_parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    test_parser.add_argument("--model", choices=model_choices, default=DEFAULT_MODEL, help=model_help)

    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Check baseline accuracy")
    baseline_parser.add_argument("--max-prompts", type=int, help="Limit prompts")
    baseline_parser.add_argument("--model", choices=model_choices, default=DEFAULT_MODEL, help=model_help)

    # Phase 1 command
    phase1_parser = subparsers.add_parser("phase1", help="Run Phase 1 analysis")
    phase1_parser.add_argument("--max-prompts", type=int, help="Limit prompts")
    phase1_parser.add_argument("--output", default="results/phase1", help="Output directory")
    phase1_parser.add_argument("--model", choices=model_choices, default=DEFAULT_MODEL, help=model_help)

    # Phase 2 command
    phase2_parser = subparsers.add_parser("phase2", help="Run Phase 2 ablation")
    phase2_parser.add_argument("--max-prompts", type=int, help="Limit prompts")
    phase2_parser.add_argument("--n-heads", type=int, default=5, help="Number of heads to ablate (default: 5)")
    phase2_parser.add_argument("--heads", type=str, help="Custom heads to ablate (e.g., 'L20H15,L21H11')")
    phase2_parser.add_argument("--control", action="store_true", help="Ablate bottom heads instead (control)")
    phase2_parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline run, use Phase 1 accuracy")
    phase2_parser.add_argument("--phase1-results", default="results/phase1", help="Path to Phase 1 results")
    phase2_parser.add_argument("--output", default="results/phase2", help="Output directory")
    phase2_parser.add_argument("--model", choices=model_choices, default=DEFAULT_MODEL, help=model_help)

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "phase1":
        cmd_phase1(args)
    elif args.command == "phase2":
        cmd_phase2(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
