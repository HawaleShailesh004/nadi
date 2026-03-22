# backend/test_rag.py — requires OPENAI_API_KEY and PINECONE_API_KEY in backend/.env
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
from engine.anomaly_detector import detect_drift
from engine.rag_engine import store_patient_history, analyze_with_ai

print("Building timeline...")
fhir = parse_patient_fhir(str(_BACKEND / "data" / "synthea_output" / "sarah.json"))
wearable = load_wearable_data(str(_BACKEND / "data" / "terra_mock" / "sarah_wearable.json"))
meds_df = pd.read_csv(_BACKEND / "data" / "medications.csv")
sarah_meds = meds_df[meds_df["patient_id"] == "sarah"].copy()
timeline = build_patient_timeline(fhir, wearable, sarah_meds)

print("Storing patient history in Pinecone...")
store_patient_history(timeline)

print("Detecting alerts...")
alerts = detect_drift(timeline["wearable_timeline"])
print(f"Alerts: {len(alerts)}")

print("Calling AI reasoning engine...")
result = analyze_with_ai(timeline, alerts)

print("\n=== AI ASSESSMENT ===")
print(json.dumps(result, indent=2))

assert result["risk_level"] in ["LOW", "MODERATE", "HIGH", "EMERGENCY"]
assert len(result["clinical_assessment"]) > 50, "Assessment too short"
assert len(result["recommendations"]) >= 2, "Need at least 2 recommendations"
assert "error" not in result, f"AI call failed: {result.get('error')}"

print("\nAll AI assertions passed ✓")
print(f"Risk level: {result['risk_level']}")
print(f"Primary cause: {result['primary_cause']}")
