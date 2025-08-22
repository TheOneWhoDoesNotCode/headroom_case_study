# Headroom Pretotype – Glossary (1‑Pager)

Use this page as a quick reference for stakeholders. Scope is per (market, brand, quarter) unless stated otherwise.

Context (v1 pilot): Germany (DE), oncology brands. Line‑of‑therapy segmentation (L1/2/3) is planned for v2.

- Priority Score
  - Weighted score blending size of prize (Headroom_Value) and ease of capture (Access_Score, Growth_Trend). Scaled to 0–1 for prioritization.
- KPI (Key Performance Indicator)
  - UI metrics (e.g., total headroom, average Priority Score) to track progress and prioritization.
- Total Patient Population
  - Epidemiology base (incidence = new cases in a period; prevalence = total existing cases at a point/period).
- Eligible Patients
  - Patients meeting the brand’s label/guideline criteria.
- Accessible Patients
  - Portion of eligible patients within payer coverage/reimbursement.
  - v1: Not multiplied as a count in the funnel; proxied via Access_Score in Priority ranking.
- Treated Patients (brand)
  - Patients actually on the selected brand’s therapy in the period (proxied by that brand’s units_sold).
- Adoption Ceiling (0–1)
  - % of eligible patients realistically expected to initiate therapy under optimal but practical conditions; caps Potential_Units.
- Potential_Units
  - patients_eligible × adoption_ceiling.
- Actual_Units
  - units_sold for the selected brand (proxy for patients treated on that brand in the period).
- Headroom_Units
  - max(Potential_Units − Actual_Units, 0).
- Price per Unit
  - Net price used to translate units to value (€).
- Headroom (Units/€)
  - For the selected brand and period, the non‑negative gap between potential and actual (size of prize). Value = Headroom_Units × price_per_unit.
- Access_Score (0–1)
  - Payer/reimbursement ease; higher = easier to capture headroom.
  - Contrast with Adoption Ceiling: Adoption Ceiling caps Potential_Units (volume). Access_Score does not change Potential_Units in v1; it affects Priority_Score (ordering). In v2, an explicit `accessible_fraction` can cap potential if desired.
- Growth_Trend
  - Market momentum (e.g., +0.05 = +5%); negatives clipped to 0 in scoring.

Compliance/Privacy
- Only anonymized, aggregated datasets are used. No raw PHI. Patient‑level data remain outside the prototype.

Ownership & Versioning
- Owner: Product/Analytics (Headroom)
- Version: v1.0 (Pretotype)
- Change Log: Track updates as formulas/thresholds evolve.
