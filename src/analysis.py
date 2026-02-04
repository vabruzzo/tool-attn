"""
Analysis utilities for Phase 1.

This module handles computing correlations, ranking heads,
and generating analysis outputs.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools import TOOLS, get_tool_names
from src.prompts import format_prompt_with_tools
from src.parsing import get_first_tool_name
from src.tokens import find_tool_token_spans
from src.attention import AttentionExtractor, compute_attention_entropy
from src.dataset import EvalPrompt


@dataclass
class ExperimentResult:
    """Result from running a single prompt."""
    prompt: str
    expected_tool: str
    predicted_tool: str | None
    correct: bool
    attention_to_tools: dict[str, torch.Tensor]  # tool -> (layers, heads)
    entropy: torch.Tensor  # (layers, heads)


def run_single_prompt(
    prompt: str,
    expected_tool: str,
    extractor: AttentionExtractor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tools: list[dict],
) -> ExperimentResult:
    """
    Run a single prompt and collect attention data.

    Args:
        prompt: User prompt
        expected_tool: Expected tool name
        extractor: AttentionExtractor instance
        model: HuggingFace model for generation
        tokenizer: Tokenizer
        tools: Tool definitions

    Returns:
        ExperimentResult with attention data
    """
    # Format prompt
    formatted = format_prompt_with_tools(tokenizer, prompt, tools)

    # Get token spans for tools
    tool_spans = find_tool_token_spans(tokenizer, formatted, tools)

    # Extract attention
    attention = extractor.extract_attention(formatted)

    # Generate response to get predicted tool
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    # Parse predicted tool
    predicted_tool = get_first_tool_name(response)

    # Compute attention to each tool
    attention_to_tools = extractor.get_attention_to_tools(attention, tool_spans)

    # Compute entropy
    entropy = compute_attention_entropy(attention, tool_spans)

    return ExperimentResult(
        prompt=prompt,
        expected_tool=expected_tool,
        predicted_tool=predicted_tool,
        correct=(predicted_tool == expected_tool),
        attention_to_tools=attention_to_tools,
        entropy=entropy,
    )


def run_experiment(
    dataset: list[EvalPrompt],
    model_name: str = "Qwen/Qwen3-4B",
    tools: list[dict] | None = None,
    max_prompts: int | None = None,
) -> list[ExperimentResult]:
    """
    Run the full Phase 1 experiment.

    Args:
        dataset: List of evaluation prompts
        model_name: Model to use
        tools: Tool definitions (defaults to TOOLS)
        max_prompts: Limit number of prompts (for testing)

    Returns:
        List of ExperimentResult objects
    """
    if tools is None:
        tools = TOOLS

    if max_prompts is not None:
        dataset = dataset[:max_prompts]

    # Load models
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    extractor = AttentionExtractor(model_name)
    extractor.load(use_nnsight=False)  # Use HF model directly

    results = []
    for eval_prompt in tqdm(dataset, desc="Running prompts"):
        result = run_single_prompt(
            prompt=eval_prompt.prompt,
            expected_tool=eval_prompt.expected_tool,
            extractor=extractor,
            model=model,
            tokenizer=tokenizer,
            tools=tools,
        )
        results.append(result)

    return results


def compute_head_tool_correlations(
    results: list[ExperimentResult],
    tools: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Compute correlation between attention-to-tool and tool-selected.

    For each head, compute: correlation between
    "attention to tool X" and "selected tool X" (binary).

    Args:
        results: Experiment results
        tools: Tool definitions

    Returns:
        DataFrame with columns: layer, head, tool, correlation
    """
    if tools is None:
        tools = TOOLS

    tool_names = get_tool_names(tools)

    # Get dimensions from first result
    first_attn = list(results[0].attention_to_tools.values())[0]
    n_layers, n_heads = first_attn.shape

    records = []

    for layer in range(n_layers):
        for head in range(n_heads):
            for tool_name in tool_names:
                # Collect attention values and selection indicators
                attentions = []
                selected = []

                for r in results:
                    # Attention from this head to this tool
                    attn_val = r.attention_to_tools[tool_name][layer, head].item()
                    attentions.append(attn_val)

                    # Did the model select this tool?
                    selected.append(1.0 if r.predicted_tool == tool_name else 0.0)

                # Compute correlation
                attentions = np.array(attentions)
                selected = np.array(selected)

                if np.std(attentions) > 1e-10 and np.std(selected) > 1e-10:
                    corr = np.corrcoef(attentions, selected)[0, 1]
                else:
                    corr = 0.0

                records.append({
                    "layer": layer,
                    "head": head,
                    "tool": tool_name,
                    "correlation": corr,
                    "head_id": f"L{layer}H{head}",
                })

    return pd.DataFrame(records)


def rank_heads_by_correlation(
    correlations_df: pd.DataFrame,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Rank heads by their correlation with tool selection.

    Args:
        correlations_df: DataFrame from compute_head_tool_correlations
        aggregation: How to aggregate across tools ("mean", "max", "sum")

    Returns:
        DataFrame with one row per head, ranked by correlation
    """
    # Group by head and aggregate correlations across tools
    if aggregation == "mean":
        agg_func = "mean"
    elif aggregation == "max":
        agg_func = "max"
    elif aggregation == "sum":
        agg_func = "sum"
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    ranked = correlations_df.groupby(["layer", "head", "head_id"]).agg({
        "correlation": agg_func
    }).reset_index()

    ranked = ranked.sort_values("correlation", ascending=False)
    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked


def compute_accuracy(results: list[ExperimentResult]) -> dict:
    """
    Compute accuracy metrics from results.

    Returns:
        Dict with overall and per-tool accuracy
    """
    total = len(results)
    correct = sum(1 for r in results if r.correct)

    # Per-tool accuracy
    tool_stats = {}
    for r in results:
        tool = r.expected_tool
        if tool not in tool_stats:
            tool_stats[tool] = {"total": 0, "correct": 0}
        tool_stats[tool]["total"] += 1
        if r.correct:
            tool_stats[tool]["correct"] += 1

    per_tool = {
        tool: stats["correct"] / stats["total"]
        for tool, stats in tool_stats.items()
    }

    return {
        "overall_accuracy": correct / total,
        "total_prompts": total,
        "correct_prompts": correct,
        "per_tool_accuracy": per_tool,
    }


def compute_mean_entropy(results: list[ExperimentResult]) -> torch.Tensor:
    """
    Compute mean attention entropy across all results.

    Returns:
        Tensor of shape (layers, heads)
    """
    entropies = torch.stack([r.entropy for r in results])
    return entropies.mean(dim=0)


def save_results(
    results: list[ExperimentResult],
    correlations: pd.DataFrame,
    ranked_heads: pd.DataFrame,
    accuracy: dict,
    output_dir: str | Path,
):
    """Save all analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save correlations
    correlations.to_csv(output_dir / "correlations.csv", index=False)

    # Save ranked heads
    ranked_heads.to_csv(output_dir / "ranked_heads.csv", index=False)

    # Save accuracy
    import json
    with open(output_dir / "accuracy.json", "w") as f:
        json.dump(accuracy, f, indent=2)

    # Save top heads summary
    top_20 = ranked_heads.head(20)
    with open(output_dir / "top_heads.txt", "w") as f:
        f.write("Top 20 Tool Selection Heads\n")
        f.write("=" * 40 + "\n\n")
        for _, row in top_20.iterrows():
            f.write(f"Rank {row['rank']}: {row['head_id']} (corr={row['correlation']:.4f})\n")

    print(f"Results saved to {output_dir}")


def quick_analysis(
    results: list[ExperimentResult],
    tools: list[dict] | None = None,
) -> dict:
    """
    Run quick analysis and return summary.

    Args:
        results: Experiment results
        tools: Tool definitions

    Returns:
        Summary dict
    """
    accuracy = compute_accuracy(results)
    correlations = compute_head_tool_correlations(results, tools)
    ranked = rank_heads_by_correlation(correlations)
    mean_entropy = compute_mean_entropy(results)

    top_10 = ranked.head(10)

    return {
        "accuracy": accuracy,
        "top_10_heads": top_10.to_dict("records"),
        "mean_entropy": mean_entropy.mean().item(),
        "max_correlation": ranked["correlation"].max(),
        "min_correlation": ranked["correlation"].min(),
    }
