import json
from pathlib import Path

from src.config import settings
from src.monitoring.performance_monitor import PerformanceMonitor

def main() -> None:
    monitor = PerformanceMonitor()
    metrics = monitor.run()

    Path(settings.MONITORING_REPORT_PATH).parent.mkdir(parents = True, exist_ok = True)

    with open(settings.MONITORING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent = 2)

    print("Monitoring report generated")
    print(json.dumps(metrics, indent = 2))

if __name__ == '__main__':
    main()