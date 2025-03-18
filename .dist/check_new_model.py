import mlflow
from mlflow.tracking import MlflowClient
import os

client = MlflowClient()

# Step 1: Check for the latest unregistered model version
latest_versions = client.get_latest_versions(name="BrainTumorSegmentationModel", stages=["None"])
if latest_versions:
    latest_model_version = latest_versions[0].version
    print(f"🚀 New model found: Version {latest_model_version}")

    # Step 2: Promote the new model to Production
    client.transition_model_version_stage(
        name="BrainTumorSegmentationModel",
        version=latest_model_version,
        stage="Production"
    )
    print(f"✅ Model Version {latest_model_version} moved to Production!")

    # Step 3: Restart MLflow Server
    os.system("python restart_mlflow.py")

else:
    print("No new model found.")
