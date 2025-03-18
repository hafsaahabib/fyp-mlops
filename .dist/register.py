import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all runs
runs = client.search_runs(experiment_ids=["0"])

# Check if there are any runs before accessing them
if not runs:
    print("🚨 No runs found in MLflow! Please log a model first.")
else:
    latest_run = runs[0].info.run_id  # Assuming the most recent run is the best

    # Register the model
    model_uri = f"runs:/{latest_run}/model"
    mlflow.register_model(model_uri, "BrainTumorSegmentationModel")

    print(f"✅ Model registered successfully!")
