"""
Tool call parsing utilities.

This module handles parsing tool calls from model output.
Qwen3 uses <tool_call> tags with JSON payloads.
"""

import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""
    name: str
    arguments: dict


def parse_tool_calls(output: str) -> list[ToolCall]:
    """
    Parse tool calls from Qwen3's output.

    Qwen3 uses the format:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>

    Args:
        output: Raw model output string

    Returns:
        List of parsed ToolCall objects
    """
    tool_calls = []

    # Primary pattern: Qwen3 tool call format
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, output, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            tool_calls.append(ToolCall(
                name=data.get("name", ""),
                arguments=data.get("arguments", {}),
            ))
        except json.JSONDecodeError:
            continue

    # Fallback: OpenAI-style function calling format
    if not tool_calls:
        alt_pattern = r'\{"function":\s*\{[^}]*"name":\s*"([^"]+)"'
        alt_matches = re.findall(alt_pattern, output)
        for name in alt_matches:
            tool_calls.append(ToolCall(name=name, arguments={}))

    return tool_calls


def get_first_tool_name(output: str) -> str | None:
    """
    Get the name of the first tool called in the output.

    Args:
        output: Raw model output string

    Returns:
        Tool name or None if no tool was called
    """
    calls = parse_tool_calls(output)
    return calls[0].name if calls else None


def extract_thinking(output: str) -> str | None:
    """
    Extract the thinking/reasoning block from Qwen3 output.

    Qwen3 uses <think>...</think> tags for chain-of-thought.

    Args:
        output: Raw model output string

    Returns:
        Thinking content or None if not present
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, output, re.DOTALL)
    return match.group(1).strip() if match else None
