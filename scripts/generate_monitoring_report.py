import json

from src.monitoring.monitoring_report import MonitoringReportGenerator


def main() -> None:
    generator = MonitoringReportGenerator()
    report = generator.generate_report()
    generator.save_report(report)

    print("Unified monitoring report generated.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()