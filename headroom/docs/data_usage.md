# Data Usage and Scoring Reference

This document describes the mock data schema and the formulas used by the Headroom pretotype.

## Versioning/scoping:

**in-scope:**
- v1: National-level (market/brand/quarter), CSV inputs.
  - Context: Pilot domain is Germany (DE), oncology brands.

**not in scope:**
- v2: Sub-national (territory/region) granularity; optional fields like `accessible_fraction`, `propensity_score`; potential DB pipelines (Oracle/Snowflake).

## Files and Schemas

- data/public/markets.csv
  - Columns: market, brand, quarter, patients_total, patients_eligible, adoption_ceiling, price_per_unit
  - Optional (v2): accessible_fraction (0..1) — explicit accessible stage multiplier when available
- data/public/actuals.csv
  - Columns: market, brand, quarter, units_sold, net_sales
- data/public/market_drivers.csv
  - Columns: market, brand, quarter, access_score, growth_trend, customers_reached, customers_targeted
  - Optional (v2): propensity_score — propensity‑to‑switch signal (0..1)
  - Optional (v2): engagement_score — customer engagement/coverage/response proxy (0..1)

## Join Keys

All three CSVs join on: [market, brand, quarter]

## Core Formulas

- Potential_Units = patients_eligible × adoption_ceiling
- Actual_Units = units_sold for the selected brand (proxy for patients treated on that brand in the period)
- Headroom_Units = max(Potential_Units − Actual_Units, 0)
- Headroom_Value = Headroom_Units × price_per_unit
  - v2 option: Potential_Units = patients_eligible × adoption_ceiling × accessible_fraction (if provided)

## Priority Scoring

Given config/scoring.yaml:
- weights.headroom_value (default 0.5)
- weights.access_signal (default 0.3)
- weights.reach_signal (default 0.2)
- growth_clip_min (default 0.0)

Steps:
1. HR_s = min-max scale of Headroom_Value to [0,1]
2. Sig_s = min-max scale of access_score to [0,1]
3. Reach_s = min-max scale of (customers_reached / customers_targeted) to [0,1]
   - If customers_targeted == 0, treat raw reach as 0 for safety.
4. growth_effect = clip(growth_trend, lower=growth_clip_min)
5. Priority_Score = w_hr × HR_s + w_sig × (Sig_s × (1 + growth_effect)) + w_reach × Reach_s
   - v2 option: include a propensity term, e.g., + w_prop × Prop_s, where Prop_s is scaled propensity_score
   - v2 option: include an engagement term, e.g., + w_eng × Eng_s, where Eng_s is scaled engagement_score. Configure `weights.engagement_signal` in `config/scoring.yaml`.

Note:
- Access_Score in v1 is payer/reimbursement–centric and does not include customer engagement/targeting. Engagement can be modeled via `engagement_score` in v2 and blended via the optional term above.

## Action Thresholds

- thresholds.accelerate: Priority_Score ≥ accelerate → "Accelerate"
- thresholds.nurture: Priority_Score ≥ nurture → "Nurture"
- otherwise: "Monitor"

## Formatting

- formatting.headroom_currency: currency code label for UI
- formatting.headroom_divisor: divide Headroom_Value totals for readable scale (e.g., 1_000_000)
- formatting.headroom_decimals: decimals for headroom display
- formatting.roi_decimals: decimals for Priority Score display (config key name kept as-is)

## Notes

- Missing values are handled with safe defaults (0) prior to calculations.
- Min-max scaling uses 0.5 when a series is constant to avoid divide-by-zero.
- All values are mock and for demonstration only.
 - All joins and calculations are per (market, brand, quarter). “Treated Patients” refers to patients on the selected brand’s therapy in the period (proxied by that brand’s units_sold). “Headroom” is brand- and period-specific.
 - Targeting vs. prioritization: v1 ranks opportunities (what/where). Targeting (how) in v2 can leverage propensity_score and coverage/concentration curves to estimate capture as a function of promotional reach.
