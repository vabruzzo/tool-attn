"""
Token-to-tool mapping utilities.

This module handles finding which tokens in the formatted prompt
correspond to which tool's name and description.
"""

from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class TokenSpan:
    """A span of tokens for a specific tool component."""
    tool_name: str
    start_idx: int
    end_idx: int  # exclusive
    span_type: str  # "name" or "description"

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def indices(self) -> list[int]:
        return list(range(self.start_idx, self.end_idx))


@dataclass
class ToolTokenSpans:
    """All token spans for a single tool."""
    tool_name: str
    name_span: TokenSpan | None
    desc_span: TokenSpan | None

    @property
    def all_indices(self) -> list[int]:
        """Get all token indices for this tool (name + description)."""
        indices = []
        if self.name_span:
            indices.extend(self.name_span.indices())
        if self.desc_span:
            indices.extend(self.desc_span.indices())
        return indices

    @property
    def total_tokens(self) -> int:
        """Total number of tokens for this tool."""
        return len(self.all_indices)


def _find_char_to_token_span(
    token_strs: list[str],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    """
    Map a character range to token indices.

    Args:
        token_strs: List of decoded token strings
        char_start: Start character position
        char_end: End character position

    Returns:
        Tuple of (start_token, end_token) or None if not found
    """
    char_idx = 0
    start_token = None
    end_token = None

    for i, tok_str in enumerate(token_strs):
        tok_start = char_idx
        tok_end = char_idx + len(tok_str)

        if tok_start <= char_start < tok_end and start_token is None:
            start_token = i
        if tok_start < char_end <= tok_end:
            end_token = i + 1
            break

        char_idx = tok_end

    if start_token is not None and end_token is not None:
        return (start_token, end_token)
    return None


def find_tool_token_spans(
    tokenizer: AutoTokenizer,
    formatted_prompt: str,
    tools: list[dict],
) -> list[ToolTokenSpans]:
    """
    Find the token indices corresponding to each tool's name and description.

    Args:
        tokenizer: HuggingFace tokenizer
        formatted_prompt: The formatted prompt string
        tools: List of tool definitions

    Returns:
        List of ToolTokenSpans, one per tool
    """
    # Tokenize the full prompt
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    results = []

    for tool in tools:
        name = tool["name"]
        desc = tool["description"]

        # Find name span - look for "name": "tool_name" pattern
        name_pattern = f'"name": "{name}"'
        name_start = formatted_prompt.find(name_pattern)
        name_span = None
        if name_start != -1:
            actual_name_start = name_start + len('"name": "')
            actual_name_end = actual_name_start + len(name)
            span = _find_char_to_token_span(token_strs, actual_name_start, actual_name_end)
            if span:
                name_span = TokenSpan(name, span[0], span[1], "name")

        # Find description span
        desc_start = formatted_prompt.find(desc)
        desc_span = None
        if desc_start != -1:
            span = _find_char_to_token_span(token_strs, desc_start, desc_start + len(desc))
            if span:
                desc_span = TokenSpan(name, span[0], span[1], "description")

        results.append(ToolTokenSpans(
            tool_name=name,
            name_span=name_span,
            desc_span=desc_span,
        ))

    return results


def build_tool_index_mapping(tool_spans: list[ToolTokenSpans]) -> dict[int, str]:
    """
    Build a mapping from token index to tool name.

    Args:
        tool_spans: List of ToolTokenSpans

    Returns:
        Dict mapping token index to tool name
    """
    mapping = {}
    for ts in tool_spans:
        for idx in ts.all_indices:
            mapping[idx] = ts.tool_name
    return mapping


def get_tool_token_mask(
    tool_spans: list[ToolTokenSpans],
    seq_len: int,
    tool_name: str | None = None,
) -> list[bool]:
    """
    Create a boolean mask for tool tokens.

    Args:
        tool_spans: List of ToolTokenSpans
        seq_len: Total sequence length
        tool_name: If specified, only mask tokens for this tool

    Returns:
        Boolean list of length seq_len
    """
    mask = [False] * seq_len

    for ts in tool_spans:
        if tool_name is not None and ts.tool_name != tool_name:
            continue
        for idx in ts.all_indices:
            if idx < seq_len:
                mask[idx] = True

    return mask
