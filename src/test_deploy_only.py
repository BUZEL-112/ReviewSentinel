import mlflow
import yaml
from prefect import flow
from src.orchestration.flows import deploy_aspect_model_task
from src.models.aspect_model import AspectModel
from setfit import SetFitModel

@flow(name='Test Deploy Isolation')
def test_deploy_flow(config_path='configs/pipeline_params.yaml'):
    # Initialize MLFlow experiment
    with open(config_path, 'r') as f:
        exp_name = yaml.safe_load(f).get('mlflow', {}).get('experiment_name', 'distilbert_training')
    mlflow.set_experiment(exp_name)
    
    print("Mocking a trained AspectModel...")
    model = AspectModel()
    # We load the base SetFit model so that model.save() doesn't fail on None
    model.model = SetFitModel.from_pretrained(model.model_name)
    
    print("Executing deploy_aspect_model_task...")
    deploy_aspect_model_task(model, config_path)
    print("Deploy task completed successfully!")

if __name__ == '__main__':
    test_deploy_flow()
