# Headroom Analysis Pretotype – Project Scaffolding

This repo sets up a minimal but extensible structure for a headroom analysis pretotype.
Use this as your starting point, then implement code modules in Windsurf/Claude Code.

---

## 📂 Project Structure


headroom/
├─ app/ # User interface (e.g. Streamlit, Gradio)
│ └─ app.py # Entry point for quick UI demo
│
├─ src/ # Core business + technical logic
│ ├─ compute.py # Headroom formulas, ROI scoring (to implement)
│ ├─ loaders.py # Data loading (CSV, Oracle, APIs)
│ ├─ rules.py # Business rules for recommendations
│ └─ utils.py # Helper functions
│
├─ data/
│ ├─ public/ # Mock/public demo data (CSV/Parquet)
│ │ ├─ markets.csv
│ │ ├─ actuals.csv
│ │ └─ signals.csv
│ └─ internal/ # Oracle/finance extracts (gitignored)
│
├─ config/
│ ├─ datasources.yaml # Paths/DSNs to data sources
│ └─ scoring.yaml # Configurable weights, thresholds
│
├─ docs/
│ └─ next_steps.md # Roadmap to productionization
│
├─ tests/ # Unit/integration tests
│ └─ test_compute.py
│
├─ README.md # Business framing + quickstart
├─ requirements.txt # Python dependencies
├─ .env.example # Example environment variables
└─ .gitignore # Ignore data/internal, logs, secrets

---

## 📝 README.md (starter content)

```markdown
# Headroom Analysis Pretotype

## Objective
Provide a fast, user-friendly way to consolidate market and patient data, compute headroom potential, and visualize KPIs to guide tactical planning.

## Features
- Load mock datasets (public CSVs) to simulate Oracle/finance extracts
- Compute potential, actuals, and headroom value
- Rank markets/brands by ROI score
- Visualize patient funnel & key metrics
- Provide recommended actions

## Next Steps
See `docs/next_steps.md` for how to evolve this into a production pilot.
```

```markdown
📝 docs/next_steps.md (starter content)
# Next Steps – From Pretotype to Pilot

1. **Data Integration**
   - Replace mock CSVs with Oracle finance views and Snowflake extracts.
   - Automate ingestion with n8n workflows.

2. **Automation**
   - Nightly refresh pipeline with validation & audit trail.
   - Slack/Email digest of top growth opportunities.

3. **Scoring**
   - Configurable weights in `scoring.yaml`.
   - Sensitivity testing and explainability dashboards.

4. **Governance & Compliance**
   - No raw PHI handled, only aggregated KPIs.
   - Enterprise LLM endpoints (OpenAI/Anthropic) for summaries.

5. **Scaling**
   - Containerize app and orchestrate with Airflow or similar.
   - Deploy to Pfizer’s internal environment with RBAC.

Timeline:  
- Week 0–1: Pretotype (CSV + Streamlit)  
- Week 2–3: Pilot (Oracle connectors + automation)  
- Week 4–6: Harden & scale (multi-brand rollout)
```

```text
📝 requirements.txt (starter content)
streamlit
pandas
numpy
scikit-learn
pyyaml
```

```text
📝 .gitignore (starter content)
__pycache__/
*.pyc
.env
data/internal/
state/*.db
logs/
```

👉 With this structure, you just open compute.py, app.py, and start vibe coding in Windsurf. The next_steps.md makes you look thoughtful about productionization while keeping the pretotype lean.
