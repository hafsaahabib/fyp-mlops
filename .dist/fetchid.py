from mlflow.tracking import MlflowClient
import mlflow
client = MlflowClient()

# Get all runs and print their details

# run_id = "bfd0f4ca437b482dac0628a25973d33b"
#   # Replace with the new Run ID
# model_uri = f"runs:/{run_id}/model"

# mlflow.register_model(model_uri, "BrainTumorSegmentationModel")

# print("✅ Model successfully registered!")
client.transition_model_version_stage(
    name="BrainTumorSegmentationModel",
    version="1",  # Replace with the latest version
    stage="Production"
)

print("✅ Model moved to Production!")
