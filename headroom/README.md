# Headroom Analysis Pretotype

## Objective
Provide a fast, user-friendly way to consolidate market and patient data, compute headroom potential, and visualize KPIs to guide tactical planning.

## Features
- Load mock datasets (public CSVs) to simulate Oracle/finance extracts
- Compute potential, actuals, and headroom value
- Rank markets/brands by ROI score
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

## Documentation map
- Canonical business logic spec: `docs/business_logic.md`
- Data usage & inputs: `docs/data_usage.md`
- Glossary: `docs/glossary.md`

## Next Steps
See `docs/next_steps.md` for how to evolve this into a production pilot.
