"""
Prompt formatting utilities.

This module handles formatting user messages with tool definitions
for Qwen3's chat template.
"""

from transformers import AutoTokenizer

from src.tools import tools_to_qwen_format


def format_prompt_with_tools(
    tokenizer: AutoTokenizer,
    user_message: str,
    tools: list[dict],
) -> str:
    """
    Format a user message with tools using Qwen3's chat template.

    Args:
        tokenizer: HuggingFace tokenizer for the model
        user_message: The user's query
        tools: List of tool definitions

    Returns:
        Formatted prompt string ready for the model
    """
    messages = [{"role": "user", "content": user_message}]
    qwen_tools = tools_to_qwen_format(tools)

    formatted = tokenizer.apply_chat_template(
        messages,
        tools=qwen_tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    return formatted


def format_prompts_batch(
    tokenizer: AutoTokenizer,
    user_messages: list[str],
    tools: list[dict],
) -> list[str]:
    """
    Format multiple user messages with the same tools.

    Args:
        tokenizer: HuggingFace tokenizer
        user_messages: List of user queries
        tools: List of tool definitions

    Returns:
        List of formatted prompt strings
    """
    return [
        format_prompt_with_tools(tokenizer, msg, tools)
        for msg in user_messages
    ]
