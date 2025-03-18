import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Fetch the latest model version
latest_versions = client.get_latest_versions(name="BrainTumorSegmentationModel", stages=["None"])
if latest_versions:
    latest_model_version = latest_versions[0].version
    print(f"New model found: Version {latest_model_version}")

    # Move the latest model to Production
    client.transition_model_version_stage(
        name="BrainTumorSegmentationModel",
        version=latest_model_version,
        stage="Production"
    )
    print(f"✅ Model Version {latest_model_version} moved to Production!")
else:
    print("No new model found.")
