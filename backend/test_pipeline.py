"""
Full pipeline test — run this every day.
If this breaks, you know exactly which layer broke.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Avoid UnicodeEncodeError on Windows consoles (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

from ingestion.fhir_parser import parse_patient_fhir, validate_parsed_data
from ingestion.wearable_generator import load_wearable_data
from engine.schema_mapper import build_patient_timeline

print("=" * 60)
print("CHRONOSHEALTH PIPELINE TEST")
print("=" * 60)

# ── STEP 1: Parse FHIR ────────────────────────────────────────
print("\n[1/3] Parsing FHIR data...")
try:
    fhir_data = parse_patient_fhir(str(_BACKEND / "data" / "synthea_output" / "sarah.json"))
    warnings = validate_parsed_data(fhir_data)
    print(f"  ✓ Patient: {fhir_data['name']}")
    print(f"  ✓ Conditions: {len(fhir_data['conditions'])}")
    print(f"  ✓ Medications: {len(fhir_data['medications'])}")
    print(f"  ✓ Lab results: {len(fhir_data['lab_results'])}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# ── STEP 2: Load Wearable Data ────────────────────────────────
print("\n[2/3] Loading wearable data...")
try:
    wearable = load_wearable_data(str(_BACKEND / "data" / "terra_mock" / "sarah_wearable.json"))
    records = wearable.get("data", [])
    print(f"  ✓ Days of data: {len(records)}")
    print(f"  ✓ Latest HR: {records[-1]['heart_rate_resting']} bpm")
    print(f"  ✓ Latest HRV: {records[-1]['hrv']} ms")
    print(f"  ✓ Latest Sleep: {records[-1]['sleep_efficiency']}%")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# ── STEP 3: Build Unified Timeline ───────────────────────────
print("\n[3/3] Building Unified Patient Timeline...")
try:
    meds_df = pd.read_csv(_BACKEND / "data" / "medications.csv")
    sarah_meds = meds_df[meds_df["patient_id"] == "sarah"].copy()

    timeline = build_patient_timeline(fhir_data, wearable, sarah_meds)

    print(f"  ✓ Wearable records: {len(timeline['wearable_timeline'])}")
    print(f"  ✓ Clinical events: {len(timeline['clinical_events'])}")
    print(f"  ✓ Medication events: {len(timeline['medication_events'])}")
    print(f"  ✓ Active conditions: {len(timeline['active_conditions'])}")
    print(f"  ✓ Active medications: {len(timeline['active_medications'])}")
    print(f"  ✓ Recent labs: {len(timeline['recent_labs'])}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# ── SAVE OUTPUT ───────────────────────────────────────────────
_data_dir = _BACKEND / "data"
_data_dir.mkdir(parents=True, exist_ok=True)
_timeline_path = _data_dir / "sarah_timeline.json"
with open(_timeline_path, "w", encoding="utf-8") as f:
    json.dump(timeline, f, indent=2)

print("\n" + "=" * 60)
print("ALL STEPS PASSED ✓")
print(f"Timeline saved to: {_timeline_path.relative_to(_BACKEND)}")
print("=" * 60)

# ── PRINT TIMELINE SUMMARY ────────────────────────────────────
print("\n=== ACTIVE CONDITIONS ===")
for c in timeline["active_conditions"]:
    print(f"  {c['condition']} (since {c['onset_date']})")

print("\n=== ACTIVE MEDICATIONS ===")
for m in timeline["active_medications"]:
    new_flag = " ← NEW" if m.get("is_new") else ""
    print(f"  {m['drug']} {m['dose']} — started {m['start_date']}{new_flag}")

print("\n=== LAST 5 LAB RESULTS ===")
for lab in timeline["recent_labs"][:5]:
    print(f"  {lab['test']}: {lab['value']} {lab['unit']} — {lab['date']}")

print("\n=== WEARABLE — BASELINE vs CURRENT ===")
wearable_records = timeline["wearable_timeline"]
if len(wearable_records) >= 5:
    baseline = wearable_records[:26]
    current = wearable_records[-2:]
    avg_hr = sum(r["heart_rate_resting"] for r in baseline) / len(baseline)
    avg_hrv = sum(r["hrv"] for r in baseline) / len(baseline)
    curr_hr = current[-1]["heart_rate_resting"]
    curr_hrv = current[-1]["hrv"]
    print(
        f"  HR:  baseline {avg_hr:.1f} → current {curr_hr:.1f} bpm ({((curr_hr - avg_hr) / avg_hr * 100):+.1f}%)"
    )
    print(
        f"  HRV: baseline {avg_hrv:.1f} → current {curr_hrv:.1f} ms ({((curr_hrv - avg_hrv) / avg_hrv * 100):+.1f}%)"
    )
