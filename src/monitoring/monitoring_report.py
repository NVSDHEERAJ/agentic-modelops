import json
from pathlib import Path
from typing import Any

from src.config import settings
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.drift_detector import DriftDetector

class MonitoringReportGenerator:
    def __init__(
            self,
            performance_monitor : PerformanceMonitor | None = None,
            drift_detector : DriftDetector | None = None,
    ) -> None:
        self.performance_monitor: PerformanceMonitor = (
            performance_monitor if performance_monitor is not None else PerformanceMonitor()
        )

        self.drift_detector: DriftDetector = (
            drift_detector if drift_detector is not None else DriftDetector()
        )

    def determine_status(
    self,
    performance_metrics: dict[str, Any],
    drift_report: dict[str, Any],
    baseline_metrics: dict[str, Any],
    ) -> str:
        """
        Determine system status based on:
        - Relative performance degradation vs baseline
        - Drift signals

        Returns: one of ["healthy", "watch", "warning", "critical"]
        """

        # Current metrics
        current_recall = performance_metrics["recall"]
        current_f1 = performance_metrics["f1"]

        # Baseline metrics
        baseline_recall = baseline_metrics["recall"]
        baseline_f1 = baseline_metrics["f1"]

        # Avoid division by zero
        if baseline_recall == 0 or baseline_f1 == 0:
            raise ValueError("Baseline metrics cannot be zero")

        # Relative drops
        recall_drop = (baseline_recall - current_recall) / baseline_recall
        f1_drop = (baseline_f1 - current_f1) / baseline_f1

        # Drift signals
        drift_detected = drift_report["drift_detected"]
        num_drifted_features = drift_report["num_drifted_features"]

        # ---- Decision Logic ----

        # CRITICAL: large degradation
        if recall_drop >= 0.20 or f1_drop >= 0.20:
            return "critical"

        # WARNING: moderate degradation OR high drift
        if recall_drop >= 0.10 or f1_drop >= 0.10 or num_drifted_features >= 5:
            return "warning"

        # WATCH: drift exists but performance stable
        if drift_detected:
            return "watch"

        # HEALTHY: everything stable
        return "healthy"
    
    def generate_recommendation(
            self,
            status : str
    ) -> str:
        if status == "critical":
            return (
                "Retraining recommended because model performance has degraded "
                "below the acceptable threshold."
            )
        
        if status == "warning":
            return (
                "Investigate drifted high-importance features and monitor performance closely. "
                "Retraining may be needed if degradation continues."
            )

        if status == "watch":
            return (
                "Drift detected, but performance is still acceptable. Continue monitoring "
                "before triggering retraining."
            )

        return "System healthy. Continue normal monitoring."
    
    def generate_report(self) -> dict[str, Any]:
        performance_metrics = self.performance_monitor.run()
        drift_report = self.drift_detector.run()
        with open(settings.BASELINE_METRICS_PATH, "r", encoding = "utf-8") as f:
            baseline_metrics = json.load(f)

        baseline_status = self.determine_status(
            performance_metrics=performance_metrics,
            drift_report=drift_report,
            baseline_metrics = baseline_metrics
        )

        baseline_recommendation = self.generate_recommendation(
            status = baseline_status
        )

        return {
            "baseline_status": baseline_status,
            "baseline_recommendation": baseline_recommendation,
            "baseline_metrics" : baseline_metrics,
            "performance": performance_metrics,
            "drift": drift_report,
            "agent_decision" : None
        }

    def save_report(self, report: dict[str, Any]) -> None:
        Path(settings.UNIFIED_MONITORING_REPORT_PATH).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(settings.UNIFIED_MONITORING_REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
