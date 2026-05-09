import json
from pathlib import Path

from src.config import settings
from src.monitoring.drift_detector import DriftDetector

def main() -> None:
    detector = DriftDetector()
    report = detector.run()

    Path(settings.DRIFT_REPORT_PATH).parent.mkdir(parents = True, exist_ok = True)

    with open(settings.DRIFT_REPORT_PATH, "w", encoding = "utf-8") as f:
        json.dump(report, f, indent = 2)

    print("Drift report generated")
    print(json.dumps(report, indent = 2))

if __name__ == "__main__":
    main()