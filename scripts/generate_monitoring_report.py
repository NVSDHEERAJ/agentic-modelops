import json

from src.logging.db import DatabaseManager
from src.monitoring.monitoring_report import MonitoringReportGenerator


def main() -> None:
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    generator = MonitoringReportGenerator(db_manager=db_manager)
    report = generator.generate_report()
    generator.save_report(report)

    print("Unified monitoring report generated.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
