#!/usr/bin/env python3
"""
Tool Selection Attention Analysis - Main Entry Point

Usage:
    python main.py test          # Run setup verification tests
    python main.py baseline      # Check baseline accuracy
    python main.py phase1        # Run Phase 1 analysis
    python main.py phase1 --max-prompts 50  # Quick test run
"""

import argparse
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


MODEL_NAME = "Qwen/Qwen3-4B"


def cmd_test(args):
    """Run setup verification tests."""
    print("=" * 60)
    print("Tool Selection Attention Analysis - Setup Verification")
    print("=" * 60)

    # Test 1: Prompt formatting
    print("\n[1/5] Testing prompt formatting...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        extractor = AttentionExtractor(MODEL_NAME)
        extractor.load(use_nnsight=False)
        print(f"  Model: {extractor.n_layers} layers, {extractor.n_heads} heads")

        print("  Extracting attention...")
        attention = extractor.extract_attention(formatted)
        print(f"  Attention shape: {attention.shape}")
        print("  ✓ Attention extraction works")

        print("\n  Testing generation...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16
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
    print("=" * 60)
    print("Baseline Accuracy Check")
    print("=" * 60)

    dataset = load_curated_dataset()
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16
    )

    correct = 0
    total = len(dataset)
    tool_results = {name: {"correct": 0, "total": 0} for name in get_tool_names()}

    print(f"\nRunning {total} prompts...")
    for i, eval_prompt in enumerate(dataset):
        formatted = format_prompt_with_tools(tokenizer, eval_prompt.prompt, TOOLS)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:])
        predicted = get_first_tool_name(response)

        is_correct = predicted == eval_prompt.expected_tool
        if is_correct:
            correct += 1

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


def cmd_phase1(args):
    """Run Phase 1 analysis."""
    print("=" * 60)
    print("Phase 1: Find Candidate Heads")
    print("=" * 60)

    # Load dataset
    dataset = load_curated_dataset()
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
        print(f"\nUsing {len(dataset)} prompts (limited)")
    else:
        print(f"\nUsing full dataset: {len(dataset)} prompts")

    # Run experiment
    print("\nRunning experiment...")
    results = run_experiment(
        dataset=dataset,
        model_name=MODEL_NAME,
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


def main():
    parser = argparse.ArgumentParser(
        description="Tool Selection Attention Analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run setup verification")
    test_parser.add_argument("--gpu", action="store_true", help="Run GPU tests")

    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Check baseline accuracy")
    baseline_parser.add_argument("--max-prompts", type=int, help="Limit prompts")

    # Phase 1 command
    phase1_parser = subparsers.add_parser("phase1", help="Run Phase 1 analysis")
    phase1_parser.add_argument("--max-prompts", type=int, help="Limit prompts")
    phase1_parser.add_argument("--output", default="results/phase1", help="Output directory")

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "phase1":
        cmd_phase1(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
