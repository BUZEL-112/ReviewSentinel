import argparse
import json
import os
import sys
import yaml
from pathlib import Path

# Try importing mlflow, fallback to mock if not installed
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("mlflow not installed. Exiting with error.", file=sys.stderr)
    sys.exit(1)

def load_baseline_f1(mlflow_tracking_uri: str, baseline_tag: str) -> float | None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    # Find the most recent run tagged with {baseline_tag}: true
    # We might need to search across all experiments, or a specific one.
    # For simplicity, we search all experiments or assume a default.
    experiments = client.search_experiments()
    experiment_ids = [exp.experiment_id for exp in experiments]
    
    if not experiment_ids:
        return None
        
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"tags.{baseline_tag} = 'true'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        return None
        
    run = runs.iloc[0]
    return float(run.get("metrics.eval_f1", 0.0))

def load_latest_run_f1(mlflow_tracking_uri: str, experiment_name: str) -> float:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
        
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment {experiment_name}")
        
    run = runs.iloc[0]
    return float(run.get("metrics.eval_f1", 0.0))

def evaluate_gate(new_f1: float, baseline_f1: float | None, threshold: float) -> dict:
    if baseline_f1 is None:
        return {
            "gate_passed": True,
            "new_f1": new_f1,
            "baseline_f1": None,
            "improvement": None,
            "threshold": threshold,
            "reason": "No baseline exists — first deployment auto-passes"
        }
    
    improvement = new_f1 - baseline_f1
    gate_passed = improvement >= threshold
    
    return {
        "gate_passed": gate_passed,
        "new_f1": new_f1,
        "baseline_f1": baseline_f1,
        "improvement": improvement,
        "threshold": threshold,
        "reason": (
            f"F1 improved by {improvement:.4f} (>= {threshold} required)"
            if gate_passed
            else f"F1 improvement of {improvement:.4f} is below required {threshold}"
        )
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model quality gate")
    parser.add_argument("--config", default="configs/pipeline_params.yaml")
    parser.add_argument("--f1-threshold", type=float, default=0.01)
    parser.add_argument("--baseline-tag", default="is_production")
    parser.add_argument("--output-format", choices=["github-actions", "json", "text"], default="text")
    parser.add_argument("--output-path", default="/tmp/quality-gate-report.json")
    parser.add_argument("--mode", choices=["evaluate-existing", "full-evaluation"], default="evaluate-existing")
    args = parser.parse_args()

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Read experiment name from config if available
    experiment_name = "reviewsentinel-training"
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            try:
                config = yaml.safe_load(f)
                experiment_name = config.get("mlflow", {}).get("experiment_name", experiment_name)
            except yaml.YAMLError:
                pass

    try:
        if args.mode == "full-evaluation":
            # In a real scenario, this would trigger model evaluation on latest data.
            # Here we just load the latest run as a proxy.
            print("Running full evaluation (simulated)...", file=sys.stderr)
            new_f1 = load_latest_run_f1(mlflow_tracking_uri, experiment_name)
        else:
            new_f1 = load_latest_run_f1(mlflow_tracking_uri, experiment_name)
            
        baseline_f1 = load_baseline_f1(mlflow_tracking_uri, args.baseline_tag)
        
        result = evaluate_gate(new_f1, baseline_f1, args.f1_threshold)
        
        if args.output_format == "github-actions":
            github_output = os.environ.get("GITHUB_OUTPUT")
            if github_output:
                with open(github_output, 'a') as f:
                    f.write(f"gate_passed={'true' if result['gate_passed'] else 'false'}\n")
                    f.write(f"new_f1={result['new_f1']}\n")
                    f.write(f"baseline_f1={result['baseline_f1'] if result['baseline_f1'] is not None else 'none'}\n")
            
            with open("/tmp/quality-gate-report.json", "w") as f:
                json.dump(result, f, indent=2)
                
            print(result["reason"])
            if not result["gate_passed"]:
                sys.exit(1)
                
        elif args.output_format == "json":
            with open(args.output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
            if not result["gate_passed"]:
                sys.exit(1)
                
        else:
            print(result["reason"])
            print(f"New F1: {result['new_f1']}")
            print(f"Baseline F1: {result['baseline_f1']}")
            print(f"Improvement: {result['improvement']}")
            if not result["gate_passed"]:
                sys.exit(1)
                
    except Exception as e:
        print(f"Error during quality gate evaluation: {e}", file=sys.stderr)
        # Fail open or fail closed? Usually fail closed in CI.
        sys.exit(1)

if __name__ == "__main__":
    main()
