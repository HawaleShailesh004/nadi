# FHIR parser smoke test — run from backend/ or repo root
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))
from ingestion.fhir_parser import parse_patient_fhir, validate_parsed_data

_fhir_path = _BACKEND / "data" / "synthea_output" / "sarah.json"
data = parse_patient_fhir(str(_fhir_path))

print("=== PATIENT ===")
print(f"Name: {data['name']}")
print(f"DOB: {data['birth_date']}")
print(f"Gender: {data['gender']}")

print(f"\n=== CONDITIONS ({len(data['conditions'])}) ===")
for c in data["conditions"]:
    print(f"  [{c['status']}] {c['condition']} — {c['onset_date']}")

print(f"\n=== MEDICATIONS ({len(data['medications'])}) ===")
for m in data["medications"]:
    print(f"  {m['drug']} — started {m['start_date']} — {m['status']}")

print(f"\n=== LAB RESULTS ({len(data['lab_results'])}) ===")
for lab in data["lab_results"][-5:]:
    print(f"  {lab['test']}: {lab['value']} {lab['unit']} — {lab['date']}")

print("\n=== VALIDATION ===")
warnings = validate_parsed_data(data)
if warnings:
    for w in warnings:
        print(f"  {w}")
else:
    print("  All checks passed.")
