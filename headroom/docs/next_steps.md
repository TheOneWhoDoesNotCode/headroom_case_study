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
