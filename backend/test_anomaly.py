# backend/test_anomaly.py
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

from ingestion.wearable_generator import load_wearable_data
from engine.anomaly_detector import detect_drift, get_overall_severity

# Paths relative to backend/ so this works from repo root: python backend/test_anomaly.py
_wearable_path = _BACKEND / "data" / "terra_mock" / "sarah_wearable.json"
wearable = load_wearable_data(str(_wearable_path))
timeline = wearable["data"]

alerts = detect_drift(timeline)
severity = get_overall_severity(alerts)

print(f"Overall severity: {severity}")
print(f"Alerts found: {len(alerts)}")
for a in alerts:
    print(f"\n  [{a['severity']}] {a['metric_name']}")
    print(f"  Baseline: {a['baseline_mean']} {a['unit']}")
    print(f"  Current:  {a['current_value']} {a['unit']}")
    print(f"  Drift:    {a['direction_symbol']}{a['deviation_pct']}%")

# Assertions — these must all pass
assert len(alerts) >= 2, f"Expected 2+ alerts, got {len(alerts)}"
assert severity in ["MODERATE", "HIGH", "EMERGENCY"], f"Severity too low: {severity}"
hr_alert = next((a for a in alerts if a["metric"] == "heart_rate_resting"), None)
assert hr_alert is not None, "HR alert must fire"
assert hr_alert["deviation_pct"] >= 20.0, f"HR drift too small: {hr_alert['deviation_pct']}%"
hrv_alert = next((a for a in alerts if a["metric"] == "hrv"), None)
assert hrv_alert is not None, "HRV alert must fire"

print("\nAll assertions passed ✓")
