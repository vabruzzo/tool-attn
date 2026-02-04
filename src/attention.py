"""
Attention extraction utilities.

This module handles extracting attention patterns from Qwen3
using both nnsight and transformers approaches.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel

from src.tokens import ToolTokenSpans


class AttentionExtractor:
    """
    Extract attention patterns from Qwen3.

    Supports two extraction methods:
    1. nnsight: For tracing and interventions
    2. transformers: Using output_attentions=True (more reliable)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        self.model_name = model_name
        self.model = None
        self.hf_model = None
        self.tokenizer = None
        self.n_layers = None
        self.n_heads = None
        self._device = None

    def load(self, use_nnsight: bool = True):
        """
        Load model and tokenizer.

        Args:
            use_nnsight: If True, load with nnsight wrapper for interventions
        """
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if use_nnsight:
            self.model = LanguageModel(
                self.model_name,
                device_map="auto",
                dtype=torch.bfloat16,
            )
            config = self.model.config
        else:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            config = self.hf_model.config
            self._device = next(self.hf_model.parameters()).device

        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        print(f"Loaded: {self.n_layers} layers, {self.n_heads} heads")

    def load_hf_model(self):
        """Load HuggingFace model for attention extraction (if not already loaded)."""
        if self.hf_model is not None:
            return

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self._device = next(self.hf_model.parameters()).device

    @property
    def device(self):
        if self._device is not None:
            return self._device
        if self.hf_model is not None:
            return next(self.hf_model.parameters()).device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def extract_attention(
        self,
        formatted_prompt: str,
    ) -> torch.Tensor:
        """
        Extract attention weights for a prompt.

        Uses transformers' output_attentions=True for reliable extraction.

        Args:
            formatted_prompt: The formatted prompt string

        Returns:
            Attention tensor of shape (layers, heads, seq, seq)
        """
        self.load_hf_model()

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.hf_model(**inputs, output_attentions=True)

        # Stack attention from all layers: (layers, heads, seq, seq)
        attentions = torch.stack([
            a.squeeze(0) for a in outputs.attentions
        ])

        return attentions

    def extract_attention_batch(
        self,
        formatted_prompts: list[str],
        batch_size: int = 4,
    ) -> list[torch.Tensor]:
        """
        Extract attention for multiple prompts.

        Note: Returns list rather than stacked tensor because
        sequence lengths may vary.

        Args:
            formatted_prompts: List of formatted prompt strings
            batch_size: Number of prompts to process at once

        Returns:
            List of attention tensors, one per prompt
        """
        self.load_hf_model()

        results = []
        for i in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[i:i + batch_size]

            # Process one at a time due to variable lengths
            for prompt in batch:
                attn = self.extract_attention(prompt)
                results.append(attn)

        return results

    def get_attention_to_tools(
        self,
        attention: torch.Tensor,
        tool_spans: list[ToolTokenSpans],
        query_positions: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute attention from query positions to each tool.

        Args:
            attention: Attention tensor (layers, heads, seq, seq)
            tool_spans: Token spans for each tool
            query_positions: Which positions to measure attention from.
                           If None, uses the last position.

        Returns:
            Dict mapping tool name to attention tensor (layers, heads)
        """
        n_layers, n_heads, seq_len, _ = attention.shape

        if query_positions is None:
            query_positions = [seq_len - 1]

        results = {}
        for ts in tool_spans:
            tool_indices = ts.all_indices

            if not tool_indices:
                results[ts.tool_name] = torch.zeros(n_layers, n_heads)
                continue

            # Get attention from query positions to tool tokens
            # attention[layer, head, query_pos, key_pos]
            tool_attention = attention[:, :, query_positions, :][:, :, :, tool_indices]

            # Average over query positions and tool tokens
            avg_attention = tool_attention.mean(dim=(2, 3))

            results[ts.tool_name] = avg_attention

        return results

    def get_head_tool_attention_matrix(
        self,
        attention: torch.Tensor,
        tool_spans: list[ToolTokenSpans],
        query_position: int = -1,
    ) -> torch.Tensor:
        """
        Build a matrix of attention from each head to each tool.

        Args:
            attention: Attention tensor (layers, heads, seq, seq)
            tool_spans: Token spans for each tool
            query_position: Position to measure attention from (-1 = last)

        Returns:
            Tensor of shape (layers * heads, n_tools)
        """
        n_layers, n_heads, seq_len, _ = attention.shape
        n_tools = len(tool_spans)

        if query_position < 0:
            query_position = seq_len + query_position

        # Shape: (layers * heads, n_tools)
        matrix = torch.zeros(n_layers * n_heads, n_tools)

        for tool_idx, ts in enumerate(tool_spans):
            tool_indices = ts.all_indices
            if not tool_indices:
                continue

            # attention[:, :, query_position, tool_indices].mean(dim=-1)
            # Shape: (layers, heads)
            tool_attn = attention[:, :, query_position, tool_indices].mean(dim=-1)

            # Flatten to (layers * heads,)
            matrix[:, tool_idx] = tool_attn.flatten()

        return matrix


def compute_attention_entropy(
    attention: torch.Tensor,
    tool_spans: list[ToolTokenSpans],
    query_position: int = -1,
) -> torch.Tensor:
    """
    Compute entropy of attention distribution over tools.

    Higher entropy = attention more spread out across tools.

    Args:
        attention: Attention tensor (layers, heads, seq, seq)
        tool_spans: Token spans for each tool
        query_position: Position to measure from

    Returns:
        Entropy tensor of shape (layers, heads)
    """
    n_layers, n_heads, seq_len, _ = attention.shape

    if query_position < 0:
        query_position = seq_len + query_position

    # Get attention to each tool
    tool_attentions = []
    for ts in tool_spans:
        indices = ts.all_indices
        if indices:
            tool_attn = attention[:, :, query_position, indices].sum(dim=-1)
        else:
            tool_attn = torch.zeros(n_layers, n_heads)
        tool_attentions.append(tool_attn)

    # Stack: (n_tools, layers, heads)
    tool_attentions = torch.stack(tool_attentions, dim=0)

    # Normalize to get distribution: (n_tools, layers, heads)
    tool_attentions = tool_attentions / (tool_attentions.sum(dim=0, keepdim=True) + 1e-10)

    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log(tool_attentions + 1e-10)
    entropy = -(tool_attentions * log_probs).sum(dim=0)

    return entropy  # (layers, heads)
