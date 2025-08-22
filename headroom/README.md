# Headroom Analysis Pretotype

## Objective
Provide a fast, user-friendly way to consolidate market and patient data, compute headroom potential, and visualize KPIs to guide tactical planning.

## Features
- Load mock datasets (public CSVs) to simulate Oracle/finance extracts
- Compute potential, actuals, and headroom value
- Rank markets/brands by Priority Score
- Visualize patient funnel & key metrics
- Provide recommended actions

## Quickstart

1. Create a virtual environment and install dependencies (Windows PowerShell):

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run headroom/app/app.py
```

3. LLM setup (optional for AI Insights/Brainstorm):

```bash
# .env at repo root
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini   # default model used by the app
OPENAI_API_KEY=sk-...
```

- Alternatively, copy `headroom/config/llm.example.yaml` to `headroom/config/llm.yaml` and adjust:

```yaml
provider: openai
model: gpt-4o-mini
params:
  temperature: 0.2
  max_tokens: 900
```

Notes:
- The Brainstorm section has a model dropdown limited to `gpt-4o` and `gpt-4o-mini`.
- The Insights section uses the configured default (`LLM_MODEL` or `config/llm.yaml`).

## Documentation map
- Canonical business logic spec: `docs/business_logic.md`
- Data usage & inputs: `docs/data_usage.md`
- Glossary: `docs/glossary.md`

## Next Steps
See `docs/next_steps.md` for how to evolve this into a production pilot.
