"""
Schema Mapper — THE MOST CRITICAL FILE IN THE PROJECT.

This is the bridge that transforms three incompatible data formats
into one Unified Patient Timeline (UPT).

The UPT is the contract that everything else depends on.
NEVER change the top-level keys after this is working.
If you need to add fields, add them — never rename or remove.
"""

from datetime import datetime

import pandas as pd


def _csv_bool(val) -> bool:
    """Coerce CSV / pandas cell to bool (handles string 'true'/'false')."""
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def build_patient_timeline(
    fhir_data: dict,
    wearable_data: dict,
    medications_df: pd.DataFrame,
) -> dict:
    """
    Merges FHIR + Wearable + Medication data into one Unified Patient Timeline.

    Input contracts:
    - fhir_data: output of fhir_parser.parse_patient_fhir()
    - wearable_data: output of wearable_generator.load_wearable_data()
    - medications_df: pandas DataFrame from medications.csv,
                      filtered to this patient_id

    Output: The UPT dict — single source of truth for all downstream systems
    """

    # ── WEARABLE TIMELINE ─────────────────────────────────────────────────────
    # Already clean from generator — just validate and pass through
    wearable_records = wearable_data.get("data", [])

    # Validate each record has required fields
    required_wearable_fields = [
        "date",
        "heart_rate_resting",
        "hrv",
        "spo2",
        "sleep_efficiency",
    ]
    clean_wearable = []
    for record in wearable_records:
        if all(record.get(f) is not None for f in required_wearable_fields):
            clean_wearable.append(record)
        # Silently skip incomplete records

    # ── CLINICAL EVENTS ───────────────────────────────────────────────────────
    # Merge conditions and lab results into a single dated event list
    clinical_events = []

    for condition in fhir_data.get("conditions", []):
        if condition["onset_date"] not in ("unknown", None):
            clinical_events.append(
                {
                    "date": condition["onset_date"],
                    "event_type": "diagnosis",
                    "title": condition["condition"],
                    "detail": f"Status: {condition['status']}",
                    "status": condition["status"],
                    "data_source": "fhir_clinical",
                }
            )

    for lab in fhir_data.get("lab_results", []):
        if lab["date"] not in ("unknown", None) and lab["value"] is not None:
            clinical_events.append(
                {
                    "date": lab["date"],
                    "event_type": "lab_result",
                    "title": lab["test"],
                    "detail": f"{lab['value']} {lab['unit']}",
                    "value": lab["value"],
                    "unit": lab["unit"],
                    "test_name": lab["test"],
                    "data_source": "fhir_clinical",
                }
            )

    # Sort by date
    clinical_events.sort(key=lambda x: x["date"])

    # ── MEDICATION EVENTS ─────────────────────────────────────────────────────
    medication_events = []

    for _, row in medications_df.iterrows():
        start = str(row.get("start_date", "")).strip()
        if not start or start == "nan":
            continue

        medication_events.append(
            {
                "date": start,
                "event_type": "medication_start",
                "drug": str(row.get("drug_name", "Unknown")),
                "dose": f"{row.get('dose', '?')} {row.get('unit', '')}".strip(),
                "frequency": str(row.get("frequency", "unknown")),
                "symptom_reported": str(row.get("symptom_reported", "none")),
                "is_new": _csv_bool(row.get("is_new", False)),
                "data_source": "medication_log",
            }
        )

        # If medication has a stop date, add that event too
        stop = str(row.get("stop_date", "")).strip()
        if stop and stop != "nan":
            medication_events.append(
                {
                    "date": stop,
                    "event_type": "medication_stop",
                    "drug": str(row.get("drug_name", "Unknown")),
                    "data_source": "medication_log",
                }
            )

    medication_events.sort(key=lambda x: x["date"])

    # ── ACTIVE MEDICATIONS (most recent status) ──────────────────────────────
    # Combine FHIR medications with our CSV medications
    # CSV medications take precedence (more recent, more specific)
    active_meds = []

    # From medication CSV
    for _, row in medications_df.iterrows():
        stop = str(row.get("stop_date", "")).strip()
        if not stop or stop == "nan":  # No stop date = still active
            active_meds.append(
                {
                    "drug": str(row.get("drug_name", "Unknown")),
                    "dose": f"{row.get('dose', '?')} {row.get('unit', '')}".strip(),
                    "frequency": str(row.get("frequency", "unknown")),
                    "start_date": str(row.get("start_date", "unknown")),
                    "is_new": _csv_bool(row.get("is_new", False)),
                    "source": "medication_log",
                }
            )

    # ── ACTIVE CONDITIONS ────────────────────────────────────────────────────
    active_conditions = [
        c for c in fhir_data.get("conditions", []) if c.get("status") == "active"
    ]

    # ── RECENT LABS (last 5 per test type) ──────────────────────────────────
    # Group by test name, keep most recent
    lab_by_test = {}
    for lab in fhir_data.get("lab_results", []):
        test = lab.get("test", "Unknown")
        if test not in lab_by_test:
            lab_by_test[test] = []
        lab_by_test[test].append(lab)

    recent_labs = []
    for _test_name, labs in lab_by_test.items():
        sorted_labs = sorted(labs, key=lambda x: x["date"], reverse=True)
        recent_labs.extend(sorted_labs[:2])  # 2 most recent per test type

    recent_labs.sort(key=lambda x: x["date"], reverse=True)

    # ── ASSEMBLE THE UNIFIED PATIENT TIMELINE ───────────────────────────────
    return {
        # Identity
        "patient_id": fhir_data["patient_id"],
        "patient_name": fhir_data["name"],
        "patient_gender": fhir_data["gender"],
        "patient_dob": fhir_data["birth_date"],
        "generated_at": datetime.now().isoformat(),
        # The three data streams — ordered chronologically
        "wearable_timeline": clean_wearable,
        "clinical_events": clinical_events,
        "medication_events": medication_events,
        # Pre-computed convenience fields for the AI context builder
        "active_medications": active_meds,
        "active_conditions": active_conditions,
        "recent_labs": recent_labs[:10],  # Top 10 most recent
        # Metadata
        "data_sources": {
            "wearable": wearable_data.get("device", "synthetic"),
            "clinical": "Synthea FHIR R4",
            "medications": "ChronosHealth medication log",
        },
        "wearable_days": len(clean_wearable),
    }
