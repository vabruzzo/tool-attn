# Tool Selection Attention Analysis

Investigating how attention heads in LLMs select tools from a set of available options.

## Hypothesis

Tool selection operates like retrieval: specific attention heads attend to tool descriptions based on semantic similarity to the user query. These "tool retrieval heads" can be identified, ablated, and their failure modes characterized.

## Model

- **Qwen/Qwen3-4B** (36 layers, 32 heads = 1,152 total attention heads)

## Quick Start

```bash
# Install dependencies
uv sync

# Run setup verification (no GPU required)
python main.py test

# Run with GPU tests
python main.py test --gpu

# Check baseline accuracy (requires GPU)
python main.py baseline --max-prompts 50

# Run full Phase 1 analysis
python main.py phase1
```

## Project Structure

```
tool-attn/
├── main.py              # CLI entry point
├── plan.md              # Detailed experiment plan
├── README.md            # This file
├── pyproject.toml       # Dependencies
├── src/
│   ├── __init__.py
│   ├── tools.py         # Tool definitions (10 base + 40 extra for scaling)
│   ├── prompts.py       # Prompt formatting for Qwen3
│   ├── parsing.py       # Tool call parsing from model output
│   ├── tokens.py        # Token-to-tool mapping
│   ├── attention.py     # Attention extraction utilities
│   ├── dataset.py       # 500 curated evaluation prompts
│   └── analysis.py      # Phase 1 correlation analysis
└── results/             # Output directory for analysis results
```

## Commands

| Command | Description |
|---------|-------------|
| `python main.py test` | Verify setup (tokenizer, parsing, dataset) |
| `python main.py test --gpu` | Include GPU tests (model loading, generation) |
| `python main.py baseline` | Check tool selection accuracy on dataset |
| `python main.py phase1` | Run Phase 1: find candidate attention heads |

## Experiment Phases

### Phase 1: Find Candidate Heads
- Run 500 prompts through the model
- Extract attention patterns from all 1,152 heads
- Compute correlation: "attention to tool X" vs "selected tool X"
- Rank heads by correlation strength

### Phase 2: Causal Validation
- Ablate top candidate heads
- Measure accuracy drop
- Identify "essential" heads (>10% accuracy drop when ablated)

### Phase 3: Scaling
- Test with 2, 5, 10, 20, 50 tools
- Track accuracy degradation
- Measure attention entropy increase

### Phase 4: Failure Analysis
- Categorize failures: positional bias vs semantic confusion
- Visualize attention patterns on failures

## Dataset

500 curated prompts (50 per tool) that unambiguously require specific tools:

| Tool | Example Prompt |
|------|----------------|
| get_weather | "What's the weather in Tokyo?" |
| search_web | "Find recent news about SpaceX" |
| read_file | "Show me the contents of config.yaml" |
| write_file | "Save this text to notes.txt" |
| run_code | "Execute this Python code: print('Hello')" |
| send_email | "Email john@example.com about the meeting" |
| get_calendar | "What meetings do I have today?" |
| create_reminder | "Remind me to call mom at 5pm" |
| search_database | "Query for all users created last month" |
| translate_text | "Translate 'Hello' to Spanish" |

## Key Findings

*To be updated as experiments progress.*

## Dependencies

- torch>=2.0
- transformers>=4.45
- nnsight>=0.3
- einops>=0.7
- pandas>=2.0
- matplotlib>=3.8
