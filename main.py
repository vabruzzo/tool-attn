"""
Tool Selection Attention Analysis for Qwen3-4B-Instruct

This script scaffolds the experiment to identify attention heads
responsible for tool selection.
"""

import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "location": {"type": "string", "description": "City name or coordinates"},
        },
        "required": ["location"],
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "path": {"type": "string", "description": "File path to read"},
        },
        "required": ["path"],
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
    {
        "name": "run_code",
        "description": "Execute Python code",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
        },
        "required": ["to", "subject", "body"],
    },
    {
        "name": "get_calendar",
        "description": "Get calendar events for a date",
        "parameters": {
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        },
        "required": ["date"],
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder for a specific time",
        "parameters": {
            "message": {"type": "string", "description": "Reminder message"},
            "time": {"type": "string", "description": "Time for reminder (ISO 8601)"},
        },
        "required": ["message", "time"],
    },
    {
        "name": "search_database",
        "description": "Query a database",
        "parameters": {
            "query": {"type": "string", "description": "SQL query to execute"},
        },
        "required": ["query"],
    },
    {
        "name": "translate_text",
        "description": "Translate text between languages",
        "parameters": {
            "text": {"type": "string", "description": "Text to translate"},
            "source_lang": {"type": "string", "description": "Source language code"},
            "target_lang": {"type": "string", "description": "Target language code"},
        },
        "required": ["text", "target_lang"],
    },
]


def tools_to_qwen_format(tools: list[dict]) -> list[dict]:
    """Convert tool definitions to Qwen3's expected tool format."""
    qwen_tools = []
    for tool in tools:
        qwen_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool.get("parameters", {}),
                        "required": tool.get("required", []),
                    },
                },
            }
        )
    return qwen_tools


# =============================================================================
# Prompt Formatting
# =============================================================================


def format_prompt_with_tools(
    tokenizer: AutoTokenizer,
    user_message: str,
    tools: list[dict],
) -> str:
    """Format a user message with tools using Qwen3's chat template."""
    messages = [{"role": "user", "content": user_message}]
    qwen_tools = tools_to_qwen_format(tools)

    # Qwen3-Instruct uses apply_chat_template with tools parameter
    formatted = tokenizer.apply_chat_template(
        messages,
        tools=qwen_tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    return formatted


# =============================================================================
# Tool Call Parsing
# =============================================================================


@dataclass
class ToolCall:
    name: str
    arguments: dict


def parse_tool_calls(output: str) -> list[ToolCall]:
    """
    Parse tool calls from Qwen3's output.

    Qwen3 uses a specific format for tool calls:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>
    """
    tool_calls = []

    # Pattern for Qwen3 tool call format
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, output, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            tool_calls.append(
                ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                )
            )
        except json.JSONDecodeError:
            continue

    # Also try alternative format (function calling style)
    if not tool_calls:
        # Some models use: {"function": {"name": ..., "arguments": ...}}
        alt_pattern = r'\{"function":\s*\{[^}]*"name":\s*"([^"]+)"'
        alt_matches = re.findall(alt_pattern, output)
        for name in alt_matches:
            tool_calls.append(ToolCall(name=name, arguments={}))

    return tool_calls


# =============================================================================
# Token-to-Tool Mapping
# =============================================================================


@dataclass
class TokenSpan:
    tool_name: str
    start_idx: int
    end_idx: int  # exclusive
    span_type: str  # "name" or "description"


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
            indices.extend(range(self.name_span.start_idx, self.name_span.end_idx))
        if self.desc_span:
            indices.extend(range(self.desc_span.start_idx, self.desc_span.end_idx))
        return indices


def _find_char_to_token_span(
    token_strs: list[str],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    """Map character range to token indices."""
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

    Returns a list of ToolTokenSpans objects mapping tool names to their
    token positions in the formatted prompt.
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
            # Get just the tool name part (after '"name": "')
            actual_name_start = name_start + len('"name": "')
            actual_name_end = actual_name_start + len(name)
            span = _find_char_to_token_span(
                token_strs, actual_name_start, actual_name_end
            )
            if span:
                name_span = TokenSpan(name, span[0], span[1], "name")

        # Find description span
        desc_start = formatted_prompt.find(desc)
        desc_span = None
        if desc_start != -1:
            span = _find_char_to_token_span(
                token_strs, desc_start, desc_start + len(desc)
            )
            if span:
                desc_span = TokenSpan(name, span[0], span[1], "description")

        results.append(
            ToolTokenSpans(
                tool_name=name,
                name_span=name_span,
                desc_span=desc_span,
            )
        )

    return results


# =============================================================================
# Attention Extraction
# =============================================================================


class AttentionExtractor:
    """
    Extract attention patterns from Qwen3 using nnsight.

    Note: nnsight's API may vary. This provides the general pattern;
    you may need to adjust based on what's actually accessible.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.n_layers = None
        self.n_heads = None

    def load(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load with nnsight wrapper
        self.model = LanguageModel(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Get model config for dimensions
        config = self.model.config
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        print(f"Loaded: {self.n_layers} layers, {self.n_heads} heads")

    def extract_attention(
        self,
        prompt: str,
        layers: list[int] | None = None,
    ) -> dict:
        """
        Extract attention weights for given prompt.

        Args:
            prompt: Formatted prompt string
            layers: Which layers to extract (None = all)

        Returns:
            Dict with 'attention' tensor and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if layers is None:
            layers = list(range(self.n_layers))

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Storage for attention weights
        attention_weights = {}

        # Use nnsight to trace and extract attention
        with self.model.trace(input_ids) as tracer:
            for layer_idx in layers:
                layer = self.model.model.layers[layer_idx]

                # Try to access attention weights
                # The exact attribute depends on how Qwen3 implements attention
                # Common patterns:
                # - layer.self_attn.attn_weights (if exposed)
                # - Need to hook into the attention computation

                # For now, we'll save the attention output and note that
                # we may need to modify the model to expose weights
                try:
                    # Attempt 1: Direct attention weights (may not exist)
                    attn = layer.self_attn.attn_weights.save()
                    attention_weights[layer_idx] = attn
                except AttributeError:
                    # Attempt 2: Save attention output (less useful but proves hooks work)
                    attn_out = layer.self_attn.output[0].save()
                    attention_weights[layer_idx] = ("output_only", attn_out)

        return {
            "input_ids": input_ids,
            "seq_len": seq_len,
            "attention": attention_weights,
            "layers": layers,
        }

    def extract_attention_via_forward(
        self,
        prompt: str,
    ) -> torch.Tensor | None:
        """
        Alternative: Use transformers' output_attentions=True.

        This is more reliable but doesn't use nnsight's intervention capabilities.
        Useful for initial analysis before adding interventions.
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load model directly with transformers for attention output
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Required for attention weights
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
        attentions = outputs.attentions

        # Stack into single tensor: (layers, heads, seq, seq)
        stacked = torch.stack([a.squeeze(0) for a in attentions])

        return stacked


# =============================================================================
# Test Functions
# =============================================================================


def test_prompt_formatting():
    """Test that prompt formatting works correctly."""
    print("\n" + "=" * 60)
    print("Testing prompt formatting...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    prompt = "What's the weather in Tokyo?"
    formatted = format_prompt_with_tools(tokenizer, prompt, TOOLS)

    print(f"\nOriginal prompt: {prompt}")
    print(f"\nFormatted prompt (first 500 chars):\n{formatted[:500]}...")
    print(f"\nTotal length: {len(formatted)} chars")

    # Verify tools are present
    for tool in TOOLS[:3]:
        assert tool["name"] in formatted, f"Tool {tool['name']} not found"
        assert tool["description"] in formatted, (
            f"Description for {tool['name']} not found"
        )

    print("\n✓ Prompt formatting works correctly")
    return formatted


def test_token_mapping():
    """Test token-to-tool mapping."""
    print("\n" + "=" * 60)
    print("Testing token-to-tool mapping...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    prompt = "What's the weather in Tokyo?"
    formatted = format_prompt_with_tools(tokenizer, prompt, TOOLS)

    tool_spans = find_tool_token_spans(tokenizer, formatted, TOOLS)

    print(f"\nFound spans for {len(tool_spans)} tools:")
    for ts in tool_spans:
        name_info = (
            f"[{ts.name_span.start_idx}, {ts.name_span.end_idx})"
            if ts.name_span
            else "not found"
        )
        desc_info = (
            f"[{ts.desc_span.start_idx}, {ts.desc_span.end_idx})"
            if ts.desc_span
            else "not found"
        )
        print(f"  {ts.tool_name}:")
        print(f"    name: {name_info}")
        print(f"    desc: {desc_info}")
        print(f"    all indices: {len(ts.all_indices)} tokens")

    if tool_spans and tool_spans[0].name_span:
        print("\n✓ Token mapping works")
    else:
        print("\n⚠ Some spans not found - may need to adjust mapping logic")

    return tool_spans


def test_tool_parsing():
    """Test tool call parsing."""
    print("\n" + "=" * 60)
    print("Testing tool call parsing...")
    print("=" * 60)

    # Sample Qwen3 tool call output
    sample_output = """I'll check the weather for you.

<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</tool_call>"""

    calls = parse_tool_calls(sample_output)

    print(f"\nSample output:\n{sample_output}")
    print(f"\nParsed tool calls: {calls}")

    assert len(calls) == 1
    assert calls[0].name == "get_weather"

    print("\n✓ Tool parsing works correctly")
    return calls


def test_attention_extraction():
    """Test attention extraction (requires GPU)."""
    print("\n" + "=" * 60)
    print("Testing attention extraction...")
    print("=" * 60)

    extractor = AttentionExtractor()
    extractor.load()

    tokenizer = extractor.tokenizer
    prompt = "What's the weather in Tokyo?"
    formatted = format_prompt_with_tools(tokenizer, prompt, TOOLS)

    print(f"\nTrying nnsight extraction...")
    try:
        result = extractor.extract_attention(formatted, layers=[0, 1])
        print(f"  Sequence length: {result['seq_len']}")
        print(f"  Layers extracted: {result['layers']}")
        print(f"  Attention data: {list(result['attention'].keys())}")
        print("\n✓ nnsight extraction works")
    except Exception as e:
        print(f"\n⚠ nnsight extraction failed: {e}")
        print("  This may be expected - trying transformers fallback...")

    print(f"\nTrying transformers output_attentions fallback...")
    try:
        attn = extractor.extract_attention_via_forward(formatted)
        print(f"  Attention shape: {attn.shape}")
        print(f"  (layers={attn.shape[0]}, heads={attn.shape[1]}, seq={attn.shape[2]})")
        print("\n✓ Transformers attention extraction works")
    except Exception as e:
        print(f"\n✗ Transformers extraction failed: {e}")

    return extractor


def test_generation():
    """Test that the model actually calls tools."""
    print("\n" + "=" * 60)
    print("Testing tool generation...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    prompt = "What's the weather in Tokyo?"
    formatted = format_prompt_with_tools(tokenizer, prompt, TOOLS)

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    print(f"\nGenerating response for: {prompt}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )

    print(f"\nModel response:\n{response}")

    # Parse tool calls
    calls = parse_tool_calls(response)
    print(f"\nParsed tool calls: {calls}")

    if calls and calls[0].name == "get_weather":
        print("\n✓ Model correctly called get_weather")
    else:
        print("\n⚠ Model didn't call expected tool - may need to adjust prompting")

    return response, calls


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all tests to verify setup."""
    print("Tool Selection Attention Analysis - Setup Verification")
    print("=" * 60)

    # Tests that don't require GPU/model loading
    test_prompt_formatting()
    test_token_mapping()
    test_tool_parsing()

    # Tests that require model loading (comment out if no GPU)
    print("\n" + "=" * 60)
    print("GPU-REQUIRED TESTS")
    print("=" * 60)
    print("\nUncomment the following tests when you have GPU access:")
    print("  test_attention_extraction()")
    print("  test_generation()")

    test_attention_extraction()
    test_generation()

    print("\n" + "=" * 60)
    print("Setup verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
