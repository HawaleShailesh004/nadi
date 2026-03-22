"""
Pre-run and save Sarah's AI analysis.
This is your demo safety net.
NEVER call the live AI during the actual demo presentation.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

import pandas as pd

from ingestion.fhir_parser import parse_patient_fhir
from ingestion.wearable_generator import load_wearable_data
from engine.schema_mapper import build_patient_timeline
from engine.anomaly_detector import detect_drift, get_overall_severity
from engine.rag_engine import analyze_with_ai, store_patient_history

# Build everything
fhir = parse_patient_fhir(str(_BACKEND / "data" / "synthea_output" / "sarah.json"))
wearable = load_wearable_data(str(_BACKEND / "data" / "terra_mock" / "sarah_wearable.json"))
meds_df = pd.read_csv(_BACKEND / "data" / "medications.csv")
sarah_meds = meds_df[meds_df["patient_id"] == "sarah"].copy()
timeline = build_patient_timeline(fhir, wearable, sarah_meds)

# Index for RAG (skipped if fingerprint matches; optional if Pinecone/key missing)
try:
    store_patient_history(timeline)
except Exception as e:
    print(f"  Warning: Pinecone store skipped: {e}")

alerts = detect_drift(timeline["wearable_timeline"])
ai_result = analyze_with_ai(timeline, alerts)

# Save everything
demo_package = {
    "timeline": timeline,
    "alerts": alerts,
    "overall_severity": get_overall_severity(alerts),
    "ai_result": ai_result,
}

_out = _BACKEND / "data" / "sarah_demo_cache.json"
_out.parent.mkdir(parents=True, exist_ok=True)
with open(_out, "w", encoding="utf-8") as f:
    json.dump(demo_package, f, indent=2, default=str)

print("Demo cache saved ✓")
print(f"  File: {_out}")
print(f"Risk: {ai_result['risk_level']}")
print(f"Cause: {ai_result['primary_cause']}")
print("\nThis file is your backup. If anything breaks during demo,")
print("load from data/sarah_demo_cache.json and show the cached result.")
