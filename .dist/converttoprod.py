import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(
    name="BrainTumorSegmentationModel",
    version= 4,  # Replace with actual version number
    stage="Production"
)

print("✅ Model moved to Production!")
