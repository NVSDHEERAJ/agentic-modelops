import json
from pathlib import Path

from src.config import settings
from src.logging.db import DatabaseManager
from src.monitoring.performance_monitor import PerformanceMonitor


def main() -> None:
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    active_deployment = db_manager.get_active_deployment()
    monitor = PerformanceMonitor(db_manager=db_manager)
    metrics = monitor.run()

    report = {
        "active_deployment": active_deployment,
        "performance": metrics,
    }

    if active_deployment is not None:
        report_path = (
            settings.REPORT_DIR
            / active_deployment["model_version"]
            / "monitoring_report.json"
        )
    else:
        report_path = settings.MONITORING_REPORT_PATH

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Monitoring report generated")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
