import mlflow
import yaml
from prefect import flow
from src.orchestration.flows import train_aspect_model_task, deploy_aspect_model_task

@flow(name='Test Aspect Model Isolation')
def test_aspect_flow(config_path='configs/pipeline_params.yaml'):
    # Initialize MLFlow experiment
    with open(config_path, 'r') as f:
        exp_name = yaml.safe_load(f).get('mlflow', {}).get('experiment_name', 'distilbert_training')
    mlflow.set_experiment(exp_name)
    
    # We must wrap the nested task in an active MLflow run, 
    # since train_aspect_model_task uses nested=True
    with mlflow.start_run(run_name="aspect_test_run"):
        model, metrics = train_aspect_model_task(df=None, config_path=config_path)
        deploy_aspect_model_task(model, config_path)
    
    print("Test run completed successfully!")

if __name__ == '__main__':
    test_aspect_flow()
