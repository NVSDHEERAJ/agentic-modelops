# Agentic ModelOps for Fraud Detection

![No Country for Old Models](assets/no-country-for-old-models.png)

An end-to-end agentic ModelOps system for a credit card fraud detection model. The project is designed to train a baseline model, serve it through an API, monitor production behavior, diagnose drift and degradation, retrain candidate models, evaluate them against the current champion, and eventually govern deployment promotion.

The guiding idea is simple: production data changes, so the system should observe, reason, and adapt.

## What It Does

- Trains and registers a baseline LightGBM fraud model.
- Serves predictions through FastAPI using the active deployment from SQLite.
- Logs raw production requests, predictions, labels, model version, and deployment id.
- Generates deployment-scoped performance, drift, and unified monitoring reports.
- Runs a LangGraph-style agent workflow for monitoring, diagnosis, retraining, and governance.
- Retrains candidate models using baseline raw data plus labeled production logs.
- Fits candidate preprocessing artifacts on candidate train data only.
- Evaluates candidates against the active champion model.
- Enforces deterministic promotion gates and requires human approval before deployment.
- Supports active deployment switching through a versioned deployment registry.

## ModelOps Flow

```text
Production traffic
    -> FastAPI prediction service
    -> SQLite prediction logs
    -> monitoring + drift reports
    -> monitoring agent decision
    -> diagnostic retraining agent
    -> candidate training + evaluation
    -> deployment governance agent
    -> human-gated promotion or rejection
```

## System Architecture

```text
                  +--------------------------+
                  | Production Simulator     |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  | FastAPI Inference API    |
                  | active deployment loader |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  | SQLite ModelOps Store    |
                  | deployments + logs       |
                  +------------+-------------+
                               |
           +-------------------+-------------------+
           |                                       |
           v                                       v
 +----------------------+              +----------------------+
 | Performance Monitor  |              | Drift Detector       |
 +----------+-----------+              +----------+-----------+
            |                                     |
            +------------------+------------------+
                               v
                  +--------------------------+
                  | Agentic Decision Graph   |
                  | monitor -> diagnose      |
                  | -> retrain -> govern     |
                  +--------------------------+
```

## Key Design

The system keeps deployment history and production logs in SQLite.

```text
model_deployments
- id
- model_version
- model_path
- metadata_path
- preprocessing_path
- metrics_path
- trained_at
- deployed_at
- training_data_cutoff_timestamp
- is_active
- status

prediction_logs
- id
- timestamp
- model_version
- deployment_id
- features_json
- fraud_probability
- prediction
- decision_threshold
- actual_label
```

Monitoring and drift detection filter by active `deployment_id`, so historical traffic remains available without contaminating current model reports.

## Repository Layout

```text
src/
  agents/        # Agent nodes, tools, prompts, graph state
  config/        # Project paths and settings
  logging/       # SQLite schema and prediction logging
  monitoring/    # Performance, drift, unified reports
  retraining/    # Candidate data builder, preprocessing, training/evaluation
  serving/       # FastAPI app and model service
  simulation/    # Production traffic simulator
  training/      # Baseline training, evaluation, registry

scripts/
  train_baseline.py
  simulate_production_traffic.py
  run_monitoring.py
  run_drift_detection.py
  generate_monitoring_report.py
  run_agent_graph.py
  run_full_modelops_loop.py
  approve_latest_candidate.py
```

## Artifacts

Baseline and candidate artifacts are versioned.

```text
models/current/baseline_v1/
  model.pkl
  metadata.json

models/candidates/<candidate_version>/
  model.pkl
  metadata.json
  label_encoders.pkl
  preprocessing_metadata.json

reports/metrics/<model_version>/
  metrics.json
  monitoring_report.json
  drift_report.json
  unified_monitoring_report.json

reports/candidates/<candidate_version>/
  metrics.json
  candidate_summary.json
```

Raw production requests are logged to SQLite and used for retraining once labels are available.

## Running The Project

Train and register the baseline:

```bash
python scripts/train_baseline.py
```

Start the API:

```bash
uvicorn src.serving.app:app --reload
```

Run the full local ModelOps loop:

```bash
python -m scripts.run_full_modelops_loop
```

The full loop simulates traffic, generates monitoring reports, runs the agent graph, retrains/evaluates a candidate when justified, and checks whether the latest candidate is eligible for human-approved promotion.

Manual approval can be run separately:

```bash
python scripts/approve_latest_candidate.py
```

Useful individual commands:

```bash
python scripts/simulate_production_traffic.py
python scripts/run_monitoring.py
python scripts/run_drift_detection.py
python scripts/generate_monitoring_report.py
python scripts/run_agent_graph.py
python scripts/check_logs.py
```

## Governance Behavior

The deployment governance layer uses deterministic gates before any promotion can happen.

Current promotion criteria:

```text
F1 delta     >=  0.00
Recall delta >= -0.02
PR-AUC delta >= -0.01
```

If a candidate passes the gates, the system requests human approval. If it fails, promotion is blocked even if other metrics improve. This is intentional for fraud detection, where recall regression can mean missed fraud.

Example outcome:

```text
Candidate improved F1, precision, PR-AUC, and ROC-AUC,
but recall dropped beyond the allowed threshold.
Governance blocked promotion.
```

## Agent Roles

**Monitoring Agent**  
Inspects traffic, performance, drift, and retraining metadata. Produces a structured decision: healthy, watch, diagnose, or escalate.

**Diagnostic Retraining Agent**  
Investigates drift/data quality, triggers candidate retraining when justified, evaluates the latest candidate, and decides whether the candidate is ready for governance.

**Deployment Governance Agent**  
Validates promotion gates, packages human approval requests, and blocks unsafe promotion. Actual promotion requires explicit approval.

## Notes

- Production simulation sends raw feature values to the API.
- The API logs raw request features and applies preprocessing inside the model service.
- Candidate preprocessing is fit on candidate train data only, then applied to validation.
- Candidate models own their preprocessing artifacts.
- FastAPI loads the active model from the deployment registry at startup or reload.

## Project Motto

> No country for old models. Drift happens. We adapt.

