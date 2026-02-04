"""
Tool Selection Attention Analysis

A library for analyzing attention patterns in LLM tool selection.
"""

from src.tools import TOOLS, tools_to_qwen_format
from src.prompts import format_prompt_with_tools
from src.parsing import ToolCall, parse_tool_calls
from src.tokens import TokenSpan, ToolTokenSpans, find_tool_token_spans
from src.attention import AttentionExtractor

__all__ = [
    "TOOLS",
    "tools_to_qwen_format",
    "format_prompt_with_tools",
    "ToolCall",
    "parse_tool_calls",
    "TokenSpan",
    "ToolTokenSpans",
    "find_tool_token_spans",
    "AttentionExtractor",
]
