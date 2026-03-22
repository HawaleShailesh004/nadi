"""
Anomaly Detector — detects health drifts in wearable timeseries.

Clinical basis for 20% threshold:
Remote patient monitoring literature uses µ±2σ for defining
anomalous vital sign readings. For typical adult HR and HRV
variation, sustained 20%+ deviation from personal baseline
falls within that clinically validated range.

Reference: Salem et al. (2014), Sensor Fault and Patient
Anomaly Detection in Medical Wireless Sensor Networks.
"""

import statistics
from typing import List, Optional

# The four metrics we monitor, their display names, units,
# and drift direction (UP = concerning when high, DOWN = concerning when low,
# BOTH = concerning in either direction)
MONITORED_METRICS = [
    {
        "key": "heart_rate_resting",
        "name": "Resting Heart Rate",
        "unit": "bpm",
        "concerning_direction": "UP",
        # HR spiking UP is more concerning than dropping slightly
    },
    {
        "key": "hrv",
        "name": "Heart Rate Variability",
        "unit": "ms",
        "concerning_direction": "DOWN",
        # HRV dropping DOWN signals stress/illness
    },
    {
        "key": "sleep_efficiency",
        "name": "Sleep Quality",
        "unit": "%",
        "concerning_direction": "DOWN",
        # Sleep quality dropping DOWN is concerning
    },
    {
        "key": "spo2",
        "name": "Blood Oxygen (SpO2)",
        "unit": "%",
        "concerning_direction": "DOWN",
        # SpO2 dropping DOWN is always concerning
    },
]

# Severity thresholds — how much deviation before escalating
MODERATE_THRESHOLD = 0.20  # 20% deviation = MODERATE
HIGH_THRESHOLD = 0.35  # 35% deviation = HIGH
EMERGENCY_THRESHOLD = 0.50  # 50% deviation = EMERGENCY

# SpO2 has absolute clinical thresholds that override percentage logic
SPO2_EMERGENCY_FLOOR = 90.0  # Below 90% = always EMERGENCY
SPO2_HIGH_FLOOR = 94.0  # Below 94% = always HIGH


def calculate_baseline(
    records: List[dict],
    metric_key: str,
    baseline_days: int = 26,
) -> Optional[dict]:
    """
    Calculate statistical baseline from first N days of records.

    Uses first 26 days (out of 30) as baseline.
    Last 2 days are "current" — compared against this baseline.
    2 days buffer = allows drift to compound before flagging.

    Returns None if insufficient data.
    """
    values = []
    for record in records[:baseline_days]:
        val = record.get(metric_key)
        if val is not None and isinstance(val, (int, float)):
            values.append(float(val))

    if len(values) < 5:
        # Need minimum 5 data points for a meaningful baseline
        return None

    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0

    return {
        "mean": round(mean, 2),
        "stdev": round(stdev, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "sample_size": len(values),
    }


def get_current_value(
    records: List[dict],
    metric_key: str,
    current_window: int = 2,
) -> Optional[float]:
    """
    Average of the last N days = 'current' reading.

    Why average last 2 days instead of just last day?
    Single-day readings can be noisy (bad sleep one night,
    intense exercise, etc.). 2-day average reduces false positives
    while still catching real drifts quickly.
    """
    recent = records[-current_window:]
    values = [
        float(r[metric_key])
        for r in recent
        if r.get(metric_key) is not None and isinstance(r.get(metric_key), (int, float))
    ]

    if not values:
        return None

    return round(statistics.mean(values), 2)


def classify_severity(
    metric_key: str,
    baseline_mean: float,
    current_value: float,
    deviation_pct: float,
    direction: str,
) -> str:
    """
    Classify severity of a detected drift.

    SpO2 uses absolute clinical floors (overrides % logic).
    All other metrics use percentage deviation tiers.
    """
    # SpO2: always use absolute clinical thresholds
    if metric_key == "spo2":
        if current_value < SPO2_EMERGENCY_FLOOR:
            return "EMERGENCY"
        elif current_value < SPO2_HIGH_FLOOR:
            return "HIGH"
        elif deviation_pct >= MODERATE_THRESHOLD:
            return "MODERATE"
        else:
            return "LOW"

    # All other metrics: percentage deviation tiers
    if deviation_pct >= EMERGENCY_THRESHOLD:
        return "EMERGENCY"
    elif deviation_pct >= HIGH_THRESHOLD:
        return "HIGH"
    elif deviation_pct >= MODERATE_THRESHOLD:
        return "MODERATE"
    else:
        return "LOW"


def detect_drift(
    wearable_timeline: List[dict],
    threshold_pct: float = MODERATE_THRESHOLD,
) -> List[dict]:
    """
    Main detection function.

    Takes the wearable_timeline from the Unified Patient Timeline.
    Returns list of alert dicts — empty list = no alerts.

    Each alert contains everything needed to:
    1. Display it in the UI
    2. Include it in the AI reasoning prompt
    3. Generate the PDF report
    """
    if len(wearable_timeline) < 7:
        # Need at least 7 days to establish any meaningful baseline
        return []

    alerts = []

    for metric_config in MONITORED_METRICS:
        metric_key = metric_config["key"]
        metric_name = metric_config["name"]
        unit = metric_config["unit"]
        concerning_direction = metric_config["concerning_direction"]

        # Calculate baseline from first 26 days
        baseline = calculate_baseline(wearable_timeline, metric_key)
        if baseline is None:
            continue  # Skip if not enough data

        # Get current value (average of last 2 days)
        current_value = get_current_value(wearable_timeline, metric_key)
        if current_value is None:
            continue

        # Calculate deviation
        if baseline["mean"] == 0:
            continue  # Avoid division by zero

        raw_deviation = current_value - baseline["mean"]
        deviation_pct = abs(raw_deviation) / baseline["mean"]

        # Determine direction
        direction = "UP" if raw_deviation > 0 else "DOWN"

        # Only flag if deviation exceeds threshold
        if deviation_pct < threshold_pct:
            continue

        # For directional metrics, only flag if concerning direction
        if concerning_direction != "BOTH" and direction != concerning_direction:
            # e.g. HR going DOWN slightly is not our primary concern
            # But still flag if deviation is very large (emergency level)
            if deviation_pct < HIGH_THRESHOLD:
                continue

        # Classify severity
        severity = classify_severity(
            metric_key,
            baseline["mean"],
            current_value,
            deviation_pct,
            direction,
        )

        alerts.append(
            {
                "metric": metric_key,
                "metric_name": metric_name,
                "unit": unit,
                "baseline_mean": baseline["mean"],
                "baseline_stdev": baseline["stdev"],
                "current_value": current_value,
                "raw_deviation": round(raw_deviation, 2),
                "deviation_pct": round(deviation_pct * 100, 1),
                "direction": direction,
                "direction_symbol": "↑" if direction == "UP" else "↓",
                "severity": severity,
                "concerning_direction": concerning_direction,
            }
        )

    # Sort by severity (most severe first)
    severity_order = {"EMERGENCY": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    alerts.sort(key=lambda x: severity_order.get(x["severity"], 99))

    return alerts


def get_overall_severity(alerts: List[dict]) -> str:
    """
    Returns the single highest severity across all alerts.
    This is what drives the red/yellow/green badge in the UI.
    """
    if not alerts:
        return "NONE"

    severity_order = {"EMERGENCY": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    return min(alerts, key=lambda x: severity_order.get(x["severity"], 99))["severity"]
