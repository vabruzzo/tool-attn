"""
Ablation utilities for Phase 2.

This module handles zeroing out specific attention heads
and measuring the impact on tool selection accuracy.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools import TOOLS
from src.prompts import format_prompt_with_tools
from src.parsing import get_first_tool_name
from src.dataset import EvalPrompt


@dataclass
class AblationResult:
    """Result from an ablation experiment."""
    ablated_heads: list[tuple[int, int]]  # List of (layer, head) tuples
    baseline_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    per_tool_baseline: dict[str, float]
    per_tool_ablated: dict[str, float]


def create_head_ablation_hook(
    layer_idx: int,
    head_indices: list[int],
    n_heads: int,
    head_dim: int,
) -> Callable:
    """
    Create a hook that zeros out specific attention heads.

    Args:
        layer_idx: Layer index (for identification)
        head_indices: List of head indices to ablate in this layer
        n_heads: Total number of heads
        head_dim: Dimension per head

    Returns:
        Hook function to register on attention output
    """
    def hook(module, input, output):
        # output is a tuple: (attn_output, attn_weights, past_key_value)
        # attn_output shape: (batch, seq_len, n_heads * head_dim)
        attn_output = output[0]
        batch_size, seq_len, hidden_size = attn_output.shape

        # Reshape to (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, n_heads, head_dim)

        # Zero out specified heads
        for head_idx in head_indices:
            attn_output[:, :, head_idx, :] = 0.0

        # Reshape back
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)

        # Return modified output (keep other elements unchanged)
        return (attn_output,) + output[1:]

    return hook


def ablate_heads(
    model: AutoModelForCausalLM,
    heads_to_ablate: list[tuple[int, int]],
) -> list:
    """
    Register hooks to ablate specified heads.

    Args:
        model: The model to modify
        heads_to_ablate: List of (layer, head) tuples to ablate

    Returns:
        List of hook handles (for later removal)
    """
    # Group heads by layer
    layer_heads = {}
    for layer, head in heads_to_ablate:
        if layer not in layer_heads:
            layer_heads[layer] = []
        layer_heads[layer].append(head)

    # Get model config
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    handles = []
    for layer_idx, head_indices in layer_heads.items():
        # Get the attention module for this layer
        attn_module = model.model.layers[layer_idx].self_attn

        # Create and register hook
        hook = create_head_ablation_hook(layer_idx, head_indices, n_heads, head_dim)
        handle = attn_module.register_forward_hook(hook)
        handles.append(handle)

    return handles


def remove_ablation(handles: list):
    """Remove all ablation hooks."""
    for handle in handles:
        handle.remove()


def run_with_ablation(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tools: list[dict],
    heads_to_ablate: list[tuple[int, int]] | None = None,
) -> str | None:
    """
    Run a single prompt, optionally with head ablation.

    Returns:
        Predicted tool name (or None if parsing failed)
    """
    formatted = format_prompt_with_tools(tokenizer, prompt, tools)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    # Apply ablation if specified
    handles = []
    if heads_to_ablate:
        handles = ablate_heads(model, heads_to_ablate)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["</tool_call>"],
                tokenizer=tokenizer,
            )

        response = tokenizer.decode(outputs[0][seq_len:], skip_special_tokens=False)
        return get_first_tool_name(response)
    finally:
        # Always remove hooks
        remove_ablation(handles)


def run_ablation_experiment(
    dataset: list[EvalPrompt],
    model_name: str,
    heads_to_ablate: list[tuple[int, int]],
    tools: list[dict] | None = None,
    max_prompts: int | None = None,
) -> AblationResult:
    """
    Run ablation experiment comparing baseline vs ablated accuracy.

    Args:
        dataset: Evaluation prompts
        model_name: Model to use
        heads_to_ablate: List of (layer, head) tuples to ablate
        tools: Tool definitions
        max_prompts: Limit prompts (for testing)

    Returns:
        AblationResult with accuracy comparison
    """
    if tools is None:
        tools = TOOLS

    if max_prompts:
        dataset = dataset[:max_prompts]

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Ablating {len(heads_to_ablate)} heads: {heads_to_ablate}")

    # Run baseline (no ablation)
    print("\nRunning baseline (no ablation)...")
    baseline_correct = 0
    baseline_per_tool = {}

    for ep in tqdm(dataset, desc="Baseline"):
        predicted = run_with_ablation(ep.prompt, model, tokenizer, tools, None)
        is_correct = predicted == ep.expected_tool
        if is_correct:
            baseline_correct += 1

        if ep.expected_tool not in baseline_per_tool:
            baseline_per_tool[ep.expected_tool] = {"correct": 0, "total": 0}
        baseline_per_tool[ep.expected_tool]["total"] += 1
        if is_correct:
            baseline_per_tool[ep.expected_tool]["correct"] += 1

    baseline_accuracy = baseline_correct / len(dataset)

    # Run with ablation
    print(f"\nRunning with ablation ({len(heads_to_ablate)} heads)...")
    ablated_correct = 0
    ablated_per_tool = {}

    for ep in tqdm(dataset, desc="Ablated"):
        predicted = run_with_ablation(ep.prompt, model, tokenizer, tools, heads_to_ablate)
        is_correct = predicted == ep.expected_tool
        if is_correct:
            ablated_correct += 1

        if ep.expected_tool not in ablated_per_tool:
            ablated_per_tool[ep.expected_tool] = {"correct": 0, "total": 0}
        ablated_per_tool[ep.expected_tool]["total"] += 1
        if is_correct:
            ablated_per_tool[ep.expected_tool]["correct"] += 1

    ablated_accuracy = ablated_correct / len(dataset)

    # Compute per-tool accuracies
    per_tool_baseline = {
        tool: stats["correct"] / stats["total"]
        for tool, stats in baseline_per_tool.items()
    }
    per_tool_ablated = {
        tool: stats["correct"] / stats["total"]
        for tool, stats in ablated_per_tool.items()
    }

    return AblationResult(
        ablated_heads=heads_to_ablate,
        baseline_accuracy=baseline_accuracy,
        ablated_accuracy=ablated_accuracy,
        accuracy_drop=baseline_accuracy - ablated_accuracy,
        per_tool_baseline=per_tool_baseline,
        per_tool_ablated=per_tool_ablated,
    )


def load_top_heads(ranked_heads_path: str, n_heads: int = 10) -> list[tuple[int, int]]:
    """Load top N heads from Phase 1 results."""
    import pandas as pd
    df = pd.read_csv(ranked_heads_path)
    top = df.head(n_heads)
    return [(int(row["layer"]), int(row["head"])) for _, row in top.iterrows()]


def load_bottom_heads(ranked_heads_path: str, n_heads: int = 10) -> list[tuple[int, int]]:
    """Load bottom N heads from Phase 1 results (for control)."""
    import pandas as pd
    df = pd.read_csv(ranked_heads_path)
    bottom = df.tail(n_heads)
    return [(int(row["layer"]), int(row["head"])) for _, row in bottom.iterrows()]
