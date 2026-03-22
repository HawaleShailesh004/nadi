# FHIR parser smoke test — run from backend/: python test_fhir.py
import sys

sys.path.insert(0, ".")
from ingestion.fhir_parser import parse_patient_fhir, validate_parsed_data

data = parse_patient_fhir("data/synthea_output/sarah.json")

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
