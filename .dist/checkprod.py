import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Fetch versions of the registered model
versions = client.get_latest_versions("BrainTumorSegmentationModel")

for version in versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}")
