"""
Synthetic Wearable Data Generator.

Generates medically realistic 30-day wearable timeseries for a patient.
Replaces Terra API — necessary because:
1. Terra costs $399/month minimum
2. Real wearable data would never match our synthetic FHIR patient anyway
3. For a hackathon demo, we need controlled data that tells a specific story

Medical basis for the values used:
- Resting HR normal range: 60-100 bpm (AHA guidelines)
- HRV (RMSSD) normal range: 20-60ms for adults (varies by age/fitness)
- SpO2 normal range: 95-100% (below 94% = concerning)
- Sleep efficiency normal range: 75-90% (below 70% = poor sleep)
- 20% sustained deviation = clinically meaningful change (used in RPM protocols)
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_wearable_timeline(
    patient_id: str,
    days: int = 30,
    baseline_hr: float = 65.0,
    baseline_hrv: float = 48.0,
    baseline_spo2: float = 97.5,
    baseline_sleep_efficiency: float = 80.0,
    # Drift event: simulate medication starting on day 27
    drift_start_day: int = 27,
    drift_magnitude: float = 0.30,  # 30% change — exceeds our 20% threshold
) -> dict:
    """
    Generates a 30-day wearable timeline with a controlled drift event.

    Days 1-26: Normal baseline with natural day-to-day variation
    Days 27-28: Start of medication — subtle early change
    Days 29-30: Clear drift — this is what the anomaly detector catches

    The drift_start_day corresponds to when the medication was started.
    The drift becomes detectable 2 days later — medically realistic.
    """

    records = []
    start_date = datetime.now() - timedelta(days=days)

    # Natural day-to-day variation (standard deviation as % of baseline)
    # These are realistic variation ranges from clinical literature
    hr_variation = baseline_hr * 0.04  # ±4% daily variation
    hrv_variation = baseline_hrv * 0.10  # ±10% daily variation
    spo2_variation = 0.8  # ±0.8% absolute
    sleep_variation = 5.0  # ±5% absolute

    for day in range(days):
        date = start_date + timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")

        # Determine if we're in the drift zone
        days_since_drift = day - drift_start_day

        if days_since_drift < 0:
            # Normal baseline period — just natural variation
            drift_factor = 0.0
        elif days_since_drift == 0:
            # Day medication starts — very subtle change
            drift_factor = 0.05
        elif days_since_drift == 1:
            # Day 2 — change becomes visible
            drift_factor = 0.15
        else:
            # Day 3+ — full drift magnitude (30% above baseline)
            # HR goes UP with medication side effect
            # HRV goes DOWN (inverse relationship to HR)
            # Sleep goes DOWN (sympathetic activation)
            drift_factor = drift_magnitude

        # Generate each metric with natural variation + drift
        # HR: increases during medication side effect
        hr = baseline_hr + random.gauss(0, hr_variation) + (baseline_hr * drift_factor)
        hr = max(50, min(120, hr))  # Physiological bounds

        # HRV: decreases when HR increases (inverse relationship)
        hrv = baseline_hrv + random.gauss(0, hrv_variation) - (baseline_hrv * drift_factor * 0.7)
        hrv = max(10, min(100, hrv))

        # SpO2: slight decrease with medication stress
        spo2_drift = drift_factor * 1.5  # small absolute decrease
        spo2 = baseline_spo2 + random.gauss(0, spo2_variation) - spo2_drift
        spo2 = max(88, min(100, spo2))

        # Sleep efficiency: decreases with medication side effect
        sleep = (
            baseline_sleep_efficiency
            + random.gauss(0, sleep_variation)
            - (baseline_sleep_efficiency * drift_factor * 0.5)
        )
        sleep = max(40, min(98, sleep))

        records.append(
            {
                "date": date_str,
                "heart_rate_avg": round(hr + random.gauss(0, 3), 1),  # avg slightly higher than resting
                "heart_rate_resting": round(hr, 1),
                "hrv": round(hrv, 1),
                "spo2": round(spo2, 1),
                "sleep_efficiency": round(sleep, 1),
                "sleep_duration_hours": round(random.gauss(7.0, 0.8), 1),
                "steps": int(random.gauss(7500, 1500)),
                "data_source": "synthetic_wearable",
            }
        )

    return {
        "patient_id": patient_id,
        "device": "Synthetic Wearable (ChronosHealth Demo)",
        "generated_at": datetime.now().isoformat(),
        "days": days,
        "drift_start_day": drift_start_day,
        "data": records,
    }


def save_wearable_data(data: dict, output_path: str):
    """Save generated wearable data to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved wearable data to {output_path}")


def load_wearable_data(filepath: str) -> dict:
    """Load saved wearable data. Always use this in demo — never regenerate."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
