import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all registered models
models = client.search_registered_models()

# Print registered models
for model in models:
    print(f"Model Name: {model.name}")
