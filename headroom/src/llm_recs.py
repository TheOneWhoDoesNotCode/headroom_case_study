import os
import json
from typing import Any, Dict, Optional

from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file located at repo root (or nearest parent)
load_dotenv(find_dotenv(filename=".env", raise_error_if_not_found=False))

# Provider selection via env/config. For now we implement OpenAI; others can be added later.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

RECS_PROMPT = (
    "You are a senior commercial analytics strategist for oncology markets. "
    "Ground your reasoning strictly in the provided JSON data. Provide 2–3 actionable options with trade-offs. "
    "Return JSON matching the provided schema. Avoid hallucinations; cite exact metrics/quarters used."
)

INSIGHTS_PROMPT = (
    "You are a data analyst. Produce descriptive insights ONLY from the provided JSON. "
    "Requirements: (1) Quantify changes with magnitudes and % where applicable; (2) Reference exact quarters and values; "
    "(3) Highlight inflection points and variability; (4) No prescriptions, hypotheses, or advice. Be strictly data-grounded. "
    "Return JSON with: 'summary' (2–3 sentences) and 'bullets' as 3–6 concise items. Each bullet has: "
    "'text' (1 sentence, quantified, quarter-referenced) and 'data_citations' (list of quarter/metric refs)."
)

OUTPUT_SCHEMA_EXAMPLE = {
    "summary": "string",
    "options": [
        {
            "title": "string",
            "rationale": "string",
            "expected_effect": {"sales_delta": "string", "headroom_delta": "string"},
            "risks": ["string"],
            "data_citations": ["metric_or_quarter_ref"],
            "focus_alignment": 0.0,
            "why_aligned": "string"
        }
    ],
    "confidence": 0.0,
}


def _compact_float(x: Any, ndigits: int = 3) -> Any:
    try:
        return round(float(x), ndigits)
    except Exception:
        return x


def build_context_payload(view_slice, context: Dict[str, Any]) -> str:
    """Serialize a compact payload for the LLM. Expects pre-shaped context from the app.
    Only minimal shaping here to ensure stability and token efficiency.
    """
    def _round_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (d or {}).items():
            if isinstance(v, (int, float)):
                out[k] = _compact_float(v)
            else:
                out[k] = v
        return out

    payload = {
        "selection": context.get("selection", {}),
        "kpis": _round_in_dict(context.get("kpis", {})),
        "trends": context.get("trends", {}),  # assumed already trimmed to last 6–8 items
        "decomposition": context.get("decomp", None),
        "thresholds": context.get("thresholds", {}),
        "currency": context.get("currency", "EUR"),
        "notes": context.get("notes", ""),
        "schema": OUTPUT_SCHEMA_EXAMPLE,  # help steer JSON structure
    }
    return json.dumps(payload, ensure_ascii=False)


def _call_openai(messages, model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 900) -> str:
    """Call OpenAI Chat Completions with JSON output enforced."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required. Add it to requirements and install.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def llm_recommend_actions(view_slice, context: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry: build payload and call the chosen provider. Returns parsed JSON dict.
    Mode can be specified in context['mode']: 'recommendations' (default) or 'insights'.
    Temperature/max_tokens can be supplied via context['params'].
    """
    payload = build_context_payload(view_slice, context)

    mode = (context.get("mode") or "recommendations").lower()
    notes = (context.get("notes") or "").strip()

    params = context.get("params") or {}
    temperature = float(params.get("temperature", 0.2 if mode != "insights" else 0.0))
    max_tokens = int(params.get("max_tokens", 900))

    if mode == "insights":
        messages = [
            {"role": "system", "content": INSIGHTS_PROMPT},
            {"role": "user", "content": payload},
        ]
    else:
        messages = [
            {"role": "system", "content": RECS_PROMPT},
            *(
                [
                    {
                        "role": "system",
                        "content": (
                            "FOCUS: "
                            + json.dumps(notes, ensure_ascii=False)
                            + ". Prioritize options that address this focus first, quantify expected impact, and cite evidence."
                        ),
                    }
                ]
                if notes
                else []
            ),
            {"role": "user", "content": payload},
        ]

    provider = (context.get("provider") or DEFAULT_PROVIDER).lower()
    model = context.get("model") or DEFAULT_MODEL

    if provider == "openai":
        out = _call_openai(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    try:
        parsed = json.loads(out)
        # Back-compat: if insights mode returned 'options' but not 'bullets', derive bullets
        if mode == "insights":
            if isinstance(parsed, dict) and "bullets" not in parsed:
                opts = parsed.get("options") or []
                bullets = []
                for o in opts:
                    if isinstance(o, dict):
                        text = o.get("title") or o.get("rationale") or str(o)
                        cites = o.get("data_citations") or []
                        bullets.append({"text": text, "data_citations": cites})
                    else:
                        bullets.append({"text": str(o), "data_citations": []})
                parsed["bullets"] = bullets
        return parsed
    except Exception:
        # Best effort: if model returned text, wrap as a summary
        return {"summary": out, "options": [], "confidence": 0.0}
