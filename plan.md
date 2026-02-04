# Tool Selection as Retrieval: Experiment Plan

## Goal

Identify the attention heads responsible for tool selection in Qwen3-4B, validate they're causal, and find where the mechanism breaks down with many tools.

## Hypothesis

Tool selection in LLMs operates like retrieval: specific attention heads attend to tool descriptions based on semantic similarity to the user query. These "tool retrieval heads" can be identified, ablated, and their failure modes characterized.

## Model

- **Primary**: Qwen/Qwen3-4B (36 layers, 32 heads = 1,152 total heads)
- **Replication**: Qwen/Qwen3-8B (if results are interesting)

## Tool Set

10 tools with distinct purposes:

| Tool | Description |
|------|-------------|
| get_weather | Get current weather for a location |
| search_web | Search the web for information |
| read_file | Read contents of a file |
| write_file | Write content to a file |
| run_code | Execute Python code |
| send_email | Send an email to a recipient |
| get_calendar | Get calendar events for a date |
| create_reminder | Create a reminder for a specific time |
| search_database | Query a database |
| translate_text | Translate text between languages |

## Dataset

- 50 prompts per tool = 500 total prompts
- Each prompt unambiguously requires one specific tool
- Validation: >90% baseline accuracy required

---

## Phase 1: Find Candidate Heads

### Objective
Identify which attention heads correlate with tool selection.

### Method
1. For each prompt:
   - Format with all 10 tools in system prompt
   - Run forward pass, extract attention patterns from all heads
   - Record which tool was called

2. For each head:
   - Compute average attention weight to each tool's tokens
   - Compare: does high attention to tool X correlate with calling tool X?

### Metrics
- **Per-head correlation**: Pearson correlation between "attention to tool X" and "called tool X" (binary)
- **Attention entropy**: How spread out is attention across tools?

### Output
- Ranked list of heads by correlation strength
- Attention heatmaps for top candidates

### Status
- [ ] Generate 500-prompt dataset
- [ ] Validate baseline accuracy (>90%)
- [ ] Extract attention patterns for all prompts
- [ ] Compute per-head correlations
- [ ] Identify top candidate heads

---

## Phase 2: Causal Validation

### Objective
Confirm candidate heads are causally important via ablation.

### Method
1. Take top 10 heads from Phase 1
2. For each head:
   - Run all 500 prompts
   - Intervene: add large negative bias (-1e9) to attention logits for tool description tokens
   - Record: accuracy drop, which tool selected instead

### Metrics
- Selection accuracy with head ablated vs baseline
- Default behavior when ablated (positional bias? random? semantic confusion?)

### Output
- Table: head ID, baseline accuracy, ablated accuracy, delta
- "Essential" heads identified (ablation causes >10% accuracy drop)

### Status
- [ ] Implement attention intervention in nnsight
- [ ] Run ablation experiments
- [ ] Analyze default behaviors

---

## Phase 3: Scaling

### Objective
Test how tool selection degrades with more tools.

### Method
Test with tool counts: 2, 5, 10, 20, 50

For each tool count:
- Run 200 prompts (subset)
- Measure accuracy, attention entropy
- Re-run ablations on essential heads

### Output
- Plot: accuracy vs tool count
- Plot: attention entropy vs tool count
- Table: head importance at each scale

### Status
- [ ] Generate additional tools for 20+ experiments
- [ ] Run scaling experiments
- [ ] Analyze entropy trends

---

## Phase 4: Failure Mode Analysis

### Objective
Characterize why tool selection fails.

### Method
1. Filter to failed cases (wrong tool called)
2. Analyze:
   - **Positional bias**: Does it default to first/last tool?
   - **Semantic confusion**: Does it pick semantically similar wrong tool?
   - **Attention patterns**: What did attention look like on failures?

### Checks
- Shuffle tool order, re-run failures
- Compute embedding similarity between correct and chosen tools
- Visualize attention on success vs failure

### Output
- Breakdown: X% positional, Y% semantic, Z% other
- Example failure cases with visualizations

### Status
- [ ] Collect failure cases
- [ ] Implement position shuffle test
- [ ] Compute semantic similarity analysis
- [ ] Generate visualizations

---

## Project Structure

```
tool-attn/
├── main.py              # Entry point for running experiments
├── plan.md              # This file
├── README.md            # Project overview
├── pyproject.toml       # Dependencies
├── src/
│   ├── __init__.py
│   ├── tools.py         # Tool definitions
│   ├── prompts.py       # Prompt formatting utilities
│   ├── parsing.py       # Tool call parsing
│   ├── tokens.py        # Token-to-tool mapping
│   ├── attention.py     # Attention extraction
│   ├── dataset.py       # Dataset generation and loading
│   └── analysis.py      # Analysis and metrics
└── data/
    └── prompts.json     # Generated evaluation prompts
```

---

## Key Observations So Far

1. **Qwen3 uses `<think>` mode**: The model explicitly reasons about tool selection before calling. This suggests attention heads may be doing retrieval that feeds into reasoning, rather than directly selecting.

2. **Token spans confirmed**: Each tool maps to ~5-10 tokens (name + description).

3. **Both extraction methods work**:
   - nnsight: Can access attention data per layer
   - transformers: `output_attentions=True` gives full (36, 32, seq, seq) tensor

---

## Timeline

| Day | Task |
|-----|------|
| 1-2 | Setup complete, scaffold Phase 1 |
| 3 | Generate dataset, validate baseline |
| 4-5 | Extract attention, compute correlations |
| 6-7 | Phase 2: ablation experiments |
| 8-9 | Phase 3: scaling experiments |
| 10 | Phase 4: failure analysis |
| 11-12 | Plots, writeup |
