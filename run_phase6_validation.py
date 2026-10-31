"""Generate a reproducible Phase 6 validation report for the bundled datasets."""

from pathlib import Path

from src.validation import run_bundled_validation


ROOT = Path(__file__).resolve().parent
report = run_bundled_validation(ROOT)
print(f"Created {len(report['experiments'])} rigorous validation experiments.")
print("Dataset profiles:")
for profile in report["source_profiles"]:
    print(f"- {profile['dataset']}: {profile['shape'][0]} rows, {profile['shape'][1]} columns")
print("Sources were not modified.")
