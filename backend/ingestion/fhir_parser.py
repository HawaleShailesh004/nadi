"""
FHIR Parser — extracts only what ChronosHealth needs from Synthea output.
Synthea produces HL7 FHIR R4 Bundle format.
We extract: Patient demographics, Conditions, Observations (labs),
            MedicationRequests.
"""

import json
from pathlib import Path


def parse_patient_fhir(filepath: str) -> dict:
    """
    Parse a Synthea FHIR Bundle JSON file.

    Returns a clean dict with only the fields we actually use.
    Ignore everything else in the FHIR bundle — Synthea generates
    a lot of noise (encounters, care plans, etc.) we don't need.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FHIR file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    # Validate it's actually a FHIR bundle
    if bundle.get("resourceType") != "Bundle":
        raise ValueError(f"Not a FHIR Bundle: {filepath}")

    result = {
        "patient_id": None,
        "name": None,
        "birth_date": None,
        "gender": None,
        "conditions": [],
        "lab_results": [],
        "medications": [],
    }

    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")

        # ── PATIENT ──────────────────────────────────────────
        if rtype == "Patient":
            result["patient_id"] = resource.get("id", "unknown")
            result["birth_date"] = resource.get("birthDate", "unknown")
            result["gender"] = resource.get("gender", "unknown")

            # Synthea stores name as array — take first entry
            names = resource.get("name", [])
            if names:
                given = names[0].get("given", ["Unknown"])
                family = names[0].get("family", "Unknown")
                # given is a list — take first element
                first = given[0] if given else "Unknown"
                result["name"] = f"{first} {family}"

        # ── CONDITIONS (Diagnoses) ────────────────────────────
        elif rtype == "Condition":
            coding = resource.get("code", {}).get("coding", [])
            display = coding[0].get("display", "Unknown") if coding else "Unknown"

            # Get clinical status — active vs resolved
            clinical_status = (
                resource.get("clinicalStatus", {})
                .get("coding", [{}])[0]
                .get("code", "unknown")
            )

            # onset can be onsetDateTime or onsetPeriod
            onset = (
                resource.get("onsetDateTime")
                or resource.get("onsetPeriod", {}).get("start")
                or "unknown"
            )
            # Trim to date only (FHIR datetimes include time)
            if onset != "unknown" and "T" in onset:
                onset = onset.split("T")[0]

            result["conditions"].append(
                {
                    "condition": display,
                    "onset_date": onset,
                    "status": clinical_status,
                }
            )

        # ── OBSERVATIONS (Lab Results) ────────────────────────
        elif rtype == "Observation":
            # Only keep observations with a numeric value
            # Synthea also generates text observations — skip those
            value_qty = resource.get("valueQuantity")
            if not value_qty:
                continue

            coding = resource.get("code", {}).get("coding", [])
            test_name = coding[0].get("display", "Unknown") if coding else "Unknown"

            effective = (
                resource.get("effectiveDateTime")
                or resource.get("effectivePeriod", {}).get("start")
                or "unknown"
            )
            if effective != "unknown" and "T" in effective:
                effective = effective.split("T")[0]

            result["lab_results"].append(
                {
                    "test": test_name,
                    "value": value_qty.get("value"),
                    "unit": value_qty.get("unit", ""),
                    "date": effective,
                }
            )

        # ── MEDICATION REQUESTS ───────────────────────────────
        elif rtype == "MedicationRequest":
            med_coding = resource.get("medicationCodeableConcept", {}).get(
                "coding", []
            )
            drug_name = (
                med_coding[0].get("display", "Unknown") if med_coding else "Unknown"
            )

            authored = resource.get("authoredOn", "unknown")
            if authored != "unknown" and "T" in authored:
                authored = authored.split("T")[0]

            # Get dosage text if available
            dosage_instructions = resource.get("dosageInstruction", [])
            dosage_text = (
                dosage_instructions[0].get("text", "Unknown")
                if dosage_instructions
                else "Unknown"
            )

            result["medications"].append(
                {
                    "drug": drug_name,
                    "status": resource.get("status", "unknown"),
                    "start_date": authored,
                    "dosage": dosage_text,
                }
            )

    # Sort all lists by date for timeline consistency
    result["conditions"].sort(
        key=lambda x: x["onset_date"] if x["onset_date"] != "unknown" else ""
    )
    result["lab_results"].sort(
        key=lambda x: x["date"] if x["date"] != "unknown" else ""
    )
    result["medications"].sort(
        key=lambda x: x["start_date"] if x["start_date"] != "unknown" else ""
    )

    return result


def validate_parsed_data(data: dict) -> list:
    """
    Validates the parsed FHIR data.
    Returns a list of warnings — empty list means all good.
    """
    warnings = []

    if not data["patient_id"]:
        warnings.append("CRITICAL: No patient ID found")
    if not data["conditions"]:
        warnings.append("WARNING: No conditions found — pick a different patient file")
    if not data["medications"]:
        warnings.append("WARNING: No medications found — pick a different patient file")
    if len(data["lab_results"]) < 5:
        warnings.append(
            f"WARNING: Only {len(data['lab_results'])} lab results — may be insufficient"
        )

    return warnings
