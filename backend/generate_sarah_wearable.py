# backend/generate_sarah_wearable.py
import random
import sys

random.seed(42)

sys.path.insert(0, ".")
from ingestion.wearable_generator import generate_wearable_timeline, save_wearable_data

# Generate Sarah's 30-day timeline
# Drift starts day 27 (medication started 3 days ago from demo day)
data = generate_wearable_timeline(
    patient_id="sarah",
    days=30,
    baseline_hr=63.0,  # Sarah's normal resting HR
    baseline_hrv=51.0,  # Sarah's normal HRV
    baseline_spo2=97.8,  # Sarah's normal SpO2
    baseline_sleep_efficiency=81.0,
    drift_start_day=27,  # Medication started on day 27
    drift_magnitude=0.32,  # tuned so day-30 HR lands ~80–90 bpm with seed=42; still >20% drift
)

save_wearable_data(data, "data/terra_mock/sarah_wearable.json")

# Print a summary to verify
print("\n=== WEARABLE DATA SUMMARY ===")
records = data["data"]
print(f"Days generated: {len(records)}")
print("\nFirst 3 days (baseline):")
for r in records[:3]:
    print(
        f"  {r['date']}: HR={r['heart_rate_resting']} HRV={r['hrv']} Sleep={r['sleep_efficiency']}%"
    )
print("\nLast 3 days (drift zone):")
for r in records[-3:]:
    print(
        f"  {r['date']}: HR={r['heart_rate_resting']} HRV={r['hrv']} Sleep={r['sleep_efficiency']}%"
    )

# Verify the drift is detectable
baseline_hr = sum(r["heart_rate_resting"] for r in records[:26]) / 26
current_hr = records[-1]["heart_rate_resting"]
drift_pct = abs((current_hr - baseline_hr) / baseline_hr) * 100
print("\n=== DRIFT VERIFICATION ===")
print(f"Baseline HR: {baseline_hr:.1f} bpm")
print(f"Current HR: {current_hr:.1f} bpm")
status = "DETECTABLE ✓" if drift_pct >= 20 else "TOO SMALL — increase drift_magnitude"
print(f"Drift: {drift_pct:.1f}% ({status})")
