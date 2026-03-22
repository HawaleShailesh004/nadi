"""
RAG Engine — the intelligence layer of ChronosHealth.

Architecture:
1. Patient timeline + alerts → build_medical_context() → plain text context
2. Context → GPT-4o-mini with structured prompt → JSON assessment
3. Clinical events → embed_and_store() → Pinecone (for similar-pattern retrieval)
4. On query → retrieve similar past events → add to context → richer reasoning

Why RAG matters for this demo:
Without RAG: AI sees only today's numbers
With RAG: AI retrieves "last time HR spiked, what was happening?"
          and includes that in its reasoning
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

_backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(_backend_dir / ".env")
load_dotenv()

INDEX_NAME = "chronoshealth-v1"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, cheapest, fast
REASONING_MODEL = "gpt-4o-mini"  # Sufficient for structured JSON

_openai_client: Optional[OpenAI] = None
_pinecone_client: Optional[Pinecone] = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (check backend/.env)")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _get_pinecone() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        key = os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY is not set (check backend/.env)")
        _pinecone_client = Pinecone(api_key=key)
    return _pinecone_client


def _list_index_names(pc: Pinecone) -> List[str]:
    lst = pc.list_indexes()
    if hasattr(lst, "names"):
        return list(lst.names())
    return [idx.name for idx in lst]


def get_or_create_index():
    """
    Get existing index or create it.
    Called once at app startup — idempotent.
    """
    pc = _get_pinecone()
    existing = _list_index_names(pc)

    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {INDEX_NAME}")

    return pc.Index(INDEX_NAME)


def embed_text(text: str) -> List[float]:
    """
    Convert text to 1536-dim vector using OpenAI text-embedding-3-small.
    Cost: ~$0.00002 per 1K tokens. Entire demo will cost < $0.01.
    """
    response = _get_openai().embeddings.create(
        input=text[:8000],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Batch embed (OpenAI allows many inputs per request)."""
    if not texts:
        return []
    trimmed = [t[:8000] for t in texts]
    response = _get_openai().embeddings.create(
        input=trimmed,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def _env_truthy(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _pinecone_cache_path(patient_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", patient_id)[:200]
    return _backend_dir / "data" / ".pinecone_upsert_cache" / f"{safe}.sha256"


def _compute_upsert_fingerprint(patient_id: str, all_ids: List[str], all_texts: List[str]) -> str:
    """Stable hash of vector IDs + exact embed texts (same timeline => skip re-embed)."""
    lines = [f"{i}\t{t}" for i, t in zip(all_ids, all_texts)]
    raw = f"{INDEX_NAME}\n{patient_id}\n" + "\n".join(lines)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_vector_payload(timeline: dict) -> Tuple[str, List[str], List[str], List[dict]]:
    """Prepare ids, embed texts, and Pinecone metadata (no API calls)."""
    patient_id = str(timeline["patient_id"])

    clinical_items: List[tuple[str, str, dict]] = []
    for i, event in enumerate(timeline.get("clinical_events", [])):
        text = f"{event['date']}: {event['event_type']} — {event['title']}"
        if event.get("detail"):
            text += f" ({event['detail']})"
        clinical_items.append((f"{patient_id}_clinical_{i}", text, event))

    med_items: List[tuple[str, str, dict]] = []
    for i, med_event in enumerate(timeline.get("medication_events", [])):
        et = med_event.get("event_type", "medication_start")
        if et == "medication_stop":
            text = f"{med_event['date']}: stopped {med_event['drug']}"
        else:
            text = (
                f"{med_event['date']}: started {med_event['drug']} "
                f"{med_event.get('dose', '')} {med_event.get('frequency', '')}"
            ).strip()
            if med_event.get("symptom_reported") and str(
                med_event.get("symptom_reported")
            ) not in ("none", "nan", ""):
                text += f" — reported symptoms: {med_event['symptom_reported']}"
        med_items.append((f"{patient_id}_med_{i}", text, med_event))

    all_ids: List[str] = []
    all_texts: List[str] = []
    all_meta: List[dict] = []

    for vid, text, event in clinical_items:
        all_ids.append(vid)
        all_texts.append(text)
        all_meta.append(
            {
                "patient_id": patient_id,
                "date": str(event["date"]),
                "event_type": str(event["event_type"]),
                "title": str(event["title"])[:500],
                "detail": str(event.get("detail", ""))[:1000],
                "text": text[:2000],
            }
        )

    for vid, text, med_event in med_items:
        all_ids.append(vid)
        all_texts.append(text)
        is_new = med_event.get("is_new", False)
        if isinstance(is_new, str):
            is_new = is_new.strip().lower() in ("true", "1", "yes", "t")
        meta: dict[str, Any] = {
            "patient_id": patient_id,
            "date": str(med_event["date"]),
            "event_type": str(med_event.get("event_type", "medication_start")),
            "title": (
                f"Stopped {med_event['drug']}"
                if med_event.get("event_type") == "medication_stop"
                else f"Started {med_event['drug']}"
            )[:500],
            "detail": (
                f"{med_event.get('dose', '')} {med_event.get('frequency', '')}"
            ).strip()[:500],
            "text": text[:2000],
        }
        if med_event.get("event_type") != "medication_stop":
            meta["is_new"] = bool(is_new)
        all_meta.append(meta)

    return patient_id, all_ids, all_texts, all_meta


def store_patient_history(timeline: dict):
    """
    Embed and store key clinical events in Pinecone.

    This enables the RAG part: when an anomaly is detected,
    we search for semantically similar past events.

    We store: diagnoses, lab results, medication starts.
    We do NOT store: every single wearable data point (too much noise).

    Skips expensive re-embedding when the timeline payload is unchanged
    (fingerprint file under data/.pinecone_upsert_cache/). Set
    FORCE_PINECONE_UPSERT=1 to always upsert. Set SKIP_PINECONE_UPSERT=1
    to skip storage entirely (vectors must already exist in Pinecone).
    """
    patient_id, all_ids, all_texts, all_meta = _build_vector_payload(timeline)

    if _env_truthy("SKIP_PINECONE_UPSERT"):
        print(
            f"  Skipped Pinecone upsert (SKIP_PINECONE_UPSERT=1). "
            f"Using existing vectors for patient {patient_id}."
        )
        return

    if not all_texts:
        print("  Warning: no vectors to store")
        return

    fingerprint = _compute_upsert_fingerprint(patient_id, all_ids, all_texts)
    cache_path = _pinecone_cache_path(patient_id)

    if not _env_truthy("FORCE_PINECONE_UPSERT") and cache_path.is_file():
        try:
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached == fingerprint:
                print(
                    f"  Skipped Pinecone upsert (timeline unchanged, fingerprint match). "
                    f"Patient {patient_id}. Set FORCE_PINECONE_UPSERT=1 to refresh."
                )
                return
        except OSError:
            pass

    index = get_or_create_index()
    vectors: List[dict] = []
    batch_embed_size = 100

    for start in range(0, len(all_texts), batch_embed_size):
        chunk_ids = all_ids[start : start + batch_embed_size]
        chunk_texts = all_texts[start : start + batch_embed_size]
        chunk_meta = all_meta[start : start + batch_embed_size]
        try:
            embeddings = _embed_batch(chunk_texts)
        except Exception as e:
            print(f"  Warning: batch embed failed at offset {start}: {e}")
            continue
        for vid, emb, meta in zip(chunk_ids, embeddings, chunk_meta):
            vectors.append({"id": vid, "values": emb, "metadata": meta})

    if not vectors:
        print("  Warning: no vectors to store")
        return

    if len(vectors) != len(all_texts):
        print(
            f"  Warning: embedding incomplete ({len(vectors)}/{len(all_texts)}); "
            "not upserting — fix errors and retry."
        )
        return

    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=patient_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(fingerprint, encoding="utf-8")

    print(f"  Stored {len(vectors)} vectors for patient {patient_id}")


def _match_to_dict(match: Any) -> dict:
    score = getattr(match, "score", None)
    meta = getattr(match, "metadata", None) or {}
    if score is None and isinstance(match, dict):
        score = match.get("score")
        meta = match.get("metadata") or {}
    if hasattr(meta, "to_dict"):
        meta = meta.to_dict()
    return {"score": float(score) if score is not None else 0.0, "metadata": meta}


def retrieve_similar_events(
    patient_id: str,
    query_text: str,
    top_k: int = 3,
) -> List[dict]:
    """
    Retrieve most similar past events for a given query.

    Example query: "elevated heart rate medication side effect"
    Returns: past events that are semantically similar

    This is the RAG retrieval step.
    """
    index = get_or_create_index()

    try:
        query_embedding = embed_text(query_text)
        results = index.query(
            vector=query_embedding,
            top_k=max(2, top_k),
            namespace=str(patient_id),
            include_metadata=True,
        )
        matches = getattr(results, "matches", []) or []
        out: List[dict] = []
        for match in matches:
            m = _match_to_dict(match)
            score = m["score"]
            meta = m["metadata"]
            if score <= 0.5:
                continue
            out.append(
                {
                    "score": round(score, 3),
                    "text": meta.get("text", ""),
                    "date": meta.get("date", ""),
                    "event_type": meta.get("event_type", ""),
                }
            )
        return out[:top_k]
    except Exception as e:
        print(f"  Warning: retrieval failed: {e}")
        return []


def build_medical_context(
    timeline: dict,
    alerts: List[dict],
    similar_events: Optional[List[dict]] = None,
) -> str:
    """
    Builds the complete context string sent to the LLM.

    Quality of this function = quality of AI output.
    Every field here was chosen deliberately.

    Token count target: ~500-600 input tokens max.
    At gpt-4o-mini pricing, that's ~$0.0001 per call.
    """

    # Patient identity
    patient_section = (
        f"Patient: {timeline.get('patient_name', 'Unknown')}, "
        f"{timeline.get('patient_gender', 'unknown')} | "
        f"DOB: {timeline.get('patient_dob', 'unknown')}"
    )

    # Active conditions (max 5, most relevant)
    conditions = timeline.get("active_conditions", [])[:5]
    if conditions:
        conditions_text = "\n".join(
            f"- {c['condition']} (since {c.get('onset_date', 'unknown')})"
            for c in conditions
        )
    else:
        conditions_text = "- None on record"

    # Active medications — NEW medications flagged explicitly
    meds = timeline.get("active_medications", [])
    if meds:
        meds_lines = []
        for m in meds:
            flag = " ← RECENTLY STARTED" if m.get("is_new") else ""
            meds_lines.append(
                f"- {m['drug']} {m['dose']} {m['frequency']}, "
                f"since {m['start_date']}{flag}"
            )
        meds_text = "\n".join(meds_lines)
    else:
        meds_text = "- None on record"

    # Recent labs (last 4, most important)
    labs = timeline.get("recent_labs", [])[:4]
    if labs:
        labs_text = "\n".join(
            f"- {lab['test']}: {lab['value']} {lab['unit']} ({lab['date']})"
            for lab in labs
        )
    else:
        labs_text = "- None available"

    # Wearable baseline vs current
    wearable = timeline.get("wearable_timeline", [])
    if len(wearable) >= 5:
        baseline_records = wearable[:26]

        def safe_avg(records, key):
            vals = [r[key] for r in records if r.get(key) is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        b_hr = safe_avg(baseline_records, "heart_rate_resting")
        b_hrv = safe_avg(baseline_records, "hrv")
        b_sleep = safe_avg(baseline_records, "sleep_efficiency")
        b_spo2 = safe_avg(baseline_records, "spo2")

        current = wearable[-1]
        c_hr = current.get("heart_rate_resting")
        c_hrv = current.get("hrv")
        c_sleep = current.get("sleep_efficiency")
        c_spo2 = current.get("spo2")

        wearable_text = (
            f"30-day baseline → Current reading:\n"
            f"- Resting HR:    {b_hr} bpm → {c_hr} bpm\n"
            f"- HRV:           {b_hrv} ms  → {c_hrv} ms\n"
            f"- Sleep quality: {b_sleep}%  → {c_sleep}%\n"
            f"- SpO2:          {b_spo2}%  → {c_spo2}%"
        )
    else:
        wearable_text = "Insufficient wearable data"

    # Detected drifts summary
    if alerts:
        drifts_text = "\n".join(
            f"- {a['metric_name']}: {a['direction_symbol']}{a['deviation_pct']}% "
            f"from baseline [{a['severity']}]"
            for a in alerts
        )
    else:
        drifts_text = "- No significant drifts detected"

    # Similar past events from Pinecone (if available)
    if similar_events:
        similar_text = "\n".join(
            f"- [{e['date']}] {e['text']}" for e in similar_events[:3]
        )
        similar_section = f"\nRELATED PAST EVENTS (from patient history):\n{similar_text}"
    else:
        similar_section = ""

    return f"""PATIENT SUMMARY:
{patient_section}

ACTIVE DIAGNOSES:
{conditions_text}

CURRENT MEDICATIONS:
{meds_text}

RECENT LAB RESULTS:
{labs_text}

WEARABLE DATA:
{wearable_text}

DETECTED DRIFTS (>20% from 30-day baseline):
{drifts_text}{similar_section}"""


def analyze_with_ai(timeline: dict, alerts: List[dict]) -> dict:
    """
    THE CORE REASONING FUNCTION.

    Flow:
    1. Retrieve similar past events from Pinecone (RAG step)
    2. Build full medical context
    3. Send to GPT-4o-mini with structured prompt
    4. Parse and validate JSON response
    5. Return structured assessment
    """

    # If no alerts, skip AI call entirely — save cost and latency
    if not alerts:
        return {
            "risk_level": "LOW",
            "primary_cause": "No significant drifts detected",
            "clinical_assessment": (
                "All monitored vitals are within 20% of this patient's "
                "30-day personal baseline. No action required."
            ),
            "recommendations": [
                "Continue current monitoring schedule",
                "No immediate clinical action needed",
            ],
            "context_sources": ["wearable_timeline"],
            "monitor_duration_days": 0,
            "ai_model": REASONING_MODEL,
        }

    # RAG step: retrieve similar past events
    top_alert = alerts[0]
    rag_query = (
        f"{top_alert['metric_name']} {top_alert['direction']} "
        f"patient {timeline.get('patient_name', '')}"
    )
    similar_events = retrieve_similar_events(
        patient_id=str(timeline["patient_id"]),
        query_text=rag_query,
    )

    # Build context
    context = build_medical_context(timeline, alerts, similar_events)

    system_prompt = (
        "You are a clinical decision support AI. Your role is to help "
        "clinicians understand patient health drifts by analyzing integrated "
        "data from wearables, EHR records, and medication logs together.\n\n"
        "Rules:\n"
        "- ONLY reference medications, conditions, and labs explicitly listed "
        "in the patient context. Never hallucinate drug names or diagnoses.\n"
        "- If a recently started medication has known side effects matching "
        "the detected drift, state this clearly and prominently.\n"
        "- Be concise. Clinicians are busy.\n"
        "- Always respond in valid JSON."
    )

    user_prompt = f"""Analyze this patient's integrated health data and explain the detected drifts.

{context}

Respond ONLY with a JSON object in this exact format:
{{
  "risk_level": "LOW" | "MODERATE" | "HIGH" | "EMERGENCY",
  "primary_cause": "One sentence — most likely cause of the drifts",
  "clinical_assessment": "2-3 sentences — what is happening and why, grounded in the data provided",
  "recommendations": ["action 1", "action 2", "action 3"],
  "context_sources": ["list which data sources informed this assessment"],
  "monitor_duration_days": <integer, how many days to monitor before reassessing>
}}

Risk level guide:
- LOW: Expected variation, no clinical action needed
- MODERATE: Monitor closely, consider contacting prescriber
- HIGH: Contact healthcare provider within 24 hours
- EMERGENCY: Seek immediate medical care

Important: If the drift is consistent with a recently started medication
side effect, classify as MODERATE (not HIGH/EMERGENCY) unless SpO2 < 94%
or HR > 110 bpm sustained.

Metformin-related autonomic adjustment (elevated resting HR, lower HRV) is
MODERATE severity when SpO2 is normal and resting HR is under 110 bpm — not HIGH or EMERGENCY."""

    try:
        response = _get_openai().chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        result = json.loads(raw or "{}")

        required = [
            "risk_level",
            "primary_cause",
            "clinical_assessment",
            "recommendations",
            "context_sources",
            "monitor_duration_days",
        ]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing field in AI response: {field}")

        result["ai_model"] = REASONING_MODEL
        result["similar_events_used"] = len(similar_events)
        return result

    except json.JSONDecodeError as e:
        return {
            "risk_level": "MODERATE",
            "primary_cause": "Analysis incomplete — manual review recommended",
            "clinical_assessment": (
                "Automated analysis encountered an error. "
                "Please review the raw wearable data and medication log manually."
            ),
            "recommendations": ["Manual clinical review required"],
            "context_sources": ["error"],
            "monitor_duration_days": 3,
            "ai_model": REASONING_MODEL,
            "error": str(e),
        }

    except Exception as e:
        return {
            "risk_level": "MODERATE",
            "primary_cause": "Analysis service temporarily unavailable",
            "clinical_assessment": "Please retry analysis.",
            "recommendations": ["Retry in 60 seconds"],
            "context_sources": [],
            "monitor_duration_days": 1,
            "ai_model": REASONING_MODEL,
            "error": str(e),
        }
