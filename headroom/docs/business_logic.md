Business Logic: Headroom Calculation & Communication
This document captures the presentation‑ready logic behind the Headroom pretotype and how to explain it to business stakeholders.


See also: [Data Usage](data_usage.md) · [Glossary](glossary.md)

1) Purpose & Outcome
- Purpose: Identify where growth is left on the table (headroom) and what to do next per market/brand.
- Outcome: A consistent, repeatable calculation and visual funnel that shortens time‑to‑insight from weeks to minutes and aligns teams on priorities.


## Versioning/scoping:

**in-scope:**    
**v1 (This Pretotype scope):**
*   **Scope**: National-level analysis (per market/brand/quarter).
*   **Prioritization**: Ranks opportunities using Headroom (size of prize) and a blended Access/Growth score (ease of capture).
*   **Funnel**: Models 'Accessible Patients' via a proxy `Access_Score`.
*   **Data**: CSV-based inputs.
*   **Configurable**: Thresholds and weights editable in `config/scoring.yaml`.
*   **Context**: Pilot domain is Germany (DE), oncology brands.

**v2 (not in scope):**
*   **Scope**: Sub-national granularity (territory, region, HCP specialty).
*   **Segmentation**: Line-of-therapy (L1/2/3) as an optional dimension for oncology use-cases.
*   **Prioritization**: Adds a `propensity-to-switch` signal for more nuanced ranking.
*   **Funnel**: Uses an explicit `accessible_fraction` to quantify the 'Accessible Patients' stage.
*   **Targeting**: Informs 'how' to capture headroom with coverage/concentration curve analysis.
*   **Data**: Direct connection to Oracle/Snowflake pipelines.

**Beyond (Strategic Implications):**
*   **Go-to-Market (GTM) Redesign**: Use headroom insights to re-evaluate channel mix and investment allocation.
*   **Field Force Sizing & Territory Design**: Optimize sales team size and realign territory boundaries based on where the most valuable, capturable headroom is located.
*   **Lifecycle Management**: Inform indication prioritization and lifecycle planning by identifying headroom in new patient segments.


2) V1 - The Funnel at a Glance
Top → Bottom (population to sales)
a) Total Patient Population — epidemiology base (incidence/prevalence)
b) Eligible Patients — meet label/guideline criteria for the brand
c) Accessible Patients — within payer coverage/reimbursement constraints. In Germany, payer access is national, but practical accessibility may depend on prescriber uptake and hospital formularies; in v1 this is proxied by Access_Score.
d) Treated Patients — actually on the selected brand’s therapy in the period (proxied by that brand’s units sold)
e) Sales (Units × Price) — commercial output from treated units at net price
f) Headroom = Potential – Actual — compare potential treated (b × Adoption Ceiling) vs actual treated (d); size of prize in units and €, valuing with price per unit from (e)


3) Core Formulas
- Potential (patients/units)
  - Potential_Units = Eligible_Patients × Adoption_Ceiling (for the selected brand in the given market/quarter)
- Actual
  - Actual_Units = Units_Sold for the selected brand (proxy for patients treated on that brand in the period)
- Headroom
  - Headroom_Units = max(Potential_Units − Actual_Units, 0)
  - Headroom_Value = Headroom_Units × Price_Per_Unit
- ROI Score (ranking aid, configurable)
  - Normalize to 0–1 within the current dataset (using min-max scaling)
  - HR_s = scale(Headroom_Value)
  - Acc_s = scale(Access_Score)
  - Growth_effect = max(Growth_Trend, 0)
  - ROI_Score = w_headroom*HR_s + w_access*(Acc_s*(1 + Growth_effect))
  - Defaults: w_headroom=0.6, w_access=0.4

Note on granularity & joins: Calculations are per (market, brand, quarter). If `line_of_therapy` is provided (v2), extend joins to (market, brand, quarter, line_of_therapy) and interpret headroom per line.


4) Clear Definitions (glossary)
- Total Patient Population: Epidemiology base (incidence = new cases over a period; prevalence = total existing cases at a point—or period—in time).
- Eligible Patients: Patients meeting label/guideline criteria for the brand.
- Accessible Patients: Portion of eligible patients within payer coverage/reimbursement (modeled via Access_Score in v1; can be explicit in v2). In v1, Accessible Patients aren’t explicitly multiplied in the funnel but are proxied through the Access_Score adjustment in the ROI ranking.
- Treated Patients: Patients actually receiving the selected brand’s therapy in the period (proxied by that brand’s Units_Sold).
- Adoption Ceiling (0–1): % of eligible patients realistically expected to initiate therapy under optimal but practical conditions; this caps Potential_Units. It is an evidence-based input (e.g., from expert judgment or analog brand uptake), validated with cross-functional alignment (medical, market access, brand), and is set per market/brand.
- Price per Unit: Net price used to translate units to value (€).
- Headroom (Units/€): For the selected brand and period, the non‑negative gap between potential and actual (the size of prize).
- Access_Score (0–1): Payer/reimbursement ease; higher = easier to capture headroom.
- Growth_Trend: Market momentum (e.g., +0.05 = +5%); negatives are clipped to 0 in scoring.
- ROI Score: Weighted blend of size of prize and ease of capture to prioritize opportunities.
- Recommended Action: Rule‑based next step linked to ROI tier.


5) Data Contracts (tidy tables; joined on market, brand, quarter)
- markets.csv
  - Columns: market, brand, quarter, patients_total, patients_eligible, adoption_ceiling, price_per_unit
  - Optional (v2): line_of_therapy (values: L1, L2, L3) — enables per-line headroom analysis
- actuals.csv
  - Columns: market, brand, quarter, units_sold, net_sales
  - Optional (v2): line_of_therapy — aligns with markets.csv for per-line joins
- market_drivers.csv
  - Columns: market, brand, quarter, access_score, growth_trend, sentiment_score (optional)
  - Optional (v2): line_of_therapy — aligns with other tables
- Note: v1 uses public/mock data. Pilot swaps in approved Oracle/Snowflake views with the same schema — see next_steps.md.


6) Decision Rules (default)
- ROI ≥ 0.70 → Accelerate: increase call frequency + targeted content.
- 0.50 ≤ ROI < 0.70 → Nurture: digital programs + HCP webinars.
- ROI < 0.50 → Monitor: maintain, track signals.
- (Thresholds are editable in config/scoring.yaml.)


7) Example (use in slides)
- Market/Brand/Quarter: DE / Brand A / 2025Q1
- Eligible = 40,000, Adoption ceiling = 0.60 → Potential_Units = 24,000
- Actual Units_Sold = 20,000 → Headroom_Units = 4,000
- Price/Unit = €500 → Headroom_Value = €2.0M
- Access_Score = 0.75, Growth_Trend = +0.05 → ROI computed per formula → Recommended Action: Accelerate


8) Assumptions & Guardrails
- Conservative ceiling: Adoption ceiling reflects practical max, not theoretical maximum.
- No negatives: Headroom is floored at 0 (overshoot implies saturation).
- Normalization scope: Scores are normalized within the current run (dataset); show the date stamp.
- Compliance: v1 uses derived metrics only; no raw PHI. Only anonymized, aggregated datasets are used. All patient‑level data remain outside the prototype. Enterprise LLMs, if used, summarize outputs only.
- Transparency: Persist component columns (Headroom_Value, Access_Score, Growth_Trend) so rankings are explainable.
- Granularity: National‑level only in v1. Territory/region/HCP specialty cuts are planned for v2 once data are available.
- Unit consistency: Assumes `units_sold` are consistently defined across datasets (proxy for one patient treated in the period) to serve as a reliable proxy for `patients treated`.


8a) Assumptions Addendum (read‑me‑first clarifications)
- Treatment duration vs. units proxy: In v1, one unit is treated as a proxy for one patient treated in the period. For oral or cyclical therapies where units ≠ patient‑equivalents, conversion factors may be added in v2 to improve accuracy.
- Epidemiology refresh/time lag: Epidemiology inputs may lag by 1–2 years. Recent quarters may require projection/interpolation; adjustments should be version‑controlled and documented.
- Access nuance (DE context): Access_Score in v1 is a synthetic construct. In Germany, payer access is largely national; practical constraints may stem more from physician uptake or hospital formularies. This nuance can be layered in v2.
- Sentiment/competitive factor: Sentiment/competitive signals are exploratory and not included in ROI v1.
- ROI comparability: ROI scores are relative to the dataset in scope (per run). Cross‑quarter or cross‑dataset comparisons require recalibration; avoid comparing absolute ROI values across runs.


9) Communication Checklist (for slides & talk track)
- One sentence: “Headroom quantifies untapped value by comparing potential to actual; we then rank by ROI to focus where we can win fastest.”
- Show the funnel: Total → Eligible → Accessible → Treated → Sales (with the Headroom gap highlighted).
- Tie to action: End each view with Recommended Action (Accelerate/Nurture/Monitor).
- Time saved: “This standardizes the method across brands/markets and cuts manual consolidation from weeks to minutes.”
- Next step: “Swap CSVs for Oracle views; same logic, nightly refresh.”


10) Edge Cases & How We Handle Them
- Actual > Potential: Set headroom to 0; flag the row for validation (possible assumption drift).
- Epi mismatch: If epidemiology and sales diverge structurally (e.g., sustained Actual > Potential), flag for potential epidemiology assumption update.
- Zero variance in a factor (e.g., identical Access_Score): Use neutral scale value 0.5 and display a small note.
- Missing data: Exclude from ranking and surface a data quality note; allow override once validated.
- Price outliers: Cap extreme price_per_unit values or review per finance guidance.


11) What “Good” Looks Like (acceptance criteria)
- KPI cards populate and update with Market/Brand selection.
- Funnel chart and KPI table align (no contradictory counts).
- Headroom never negative; ROI ∈ [0,1].
- Prioritized table sorted by ROI; action labels match thresholds.
- Config changes (weights/thresholds) affect results without code changes.


12) Slide Layout (ready for design)
- Title: Headroom Analysis – From Population to Sales
- Layout
  - Funnel Graphic
  - Definitions Sidebar
  - Formula highlight: Headroom = Potential – Actual
  - Business Insight Footer: Cuts manual consolidation from weeks to minutes; enforces consistency across brands/markets; highlights actionable growth priorities.
- Styling tips
  - Use semi‑transparent overlays for definitions; bold the term then short definition.
  - Keep currency as €M with 1 decimal; ROI with 2 decimals.
  - Add “Data as of <YYYY‑MM‑DD>” to the app and screenshots.


13) FAQ (quick answers)
- Why not rank purely by headroom? Because ease of capture matters; access and momentum change near‑term ROI.
- Where’s ‘Accessible Patients’ in the math? Modeled via Access_Score (v1). Can be explicit as an additional multiplier in v2.
- Can we validate assumptions? Yes—add a validation layer (ranges, null checks) and a sensitivity test on weights.
- How does this scale? Same logic; swap loaders to Oracle/Snowflake and schedule nightly; see next_steps.md.
- Are we doing targeting? In v1 we prioritize markets/brands (what/where). Targeting (how) comes next: add propensity‑to‑switch signals and coverage/concentration curves to estimate capture given promotional reach (v2).


Owner: Product/Analytics (Headroom)
Version: v1.0 (Pretotype)
Change Log: Add entries as formulas/thresholds evolve.


If you want, I can also drop in a one‑page “Glossary” you can print or append as an appendix.
