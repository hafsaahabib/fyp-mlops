import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI (Ensure this is correct)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Update if needed

client = MlflowClient()

# 🔎 Fetch the latest run dynamically
experiment_id = "0"  # Default experiment ID (change if needed)
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"])

if not runs:
    print("❌ No runs found!")
    exit()

latest_run_id = runs[0].info.run_id  # Get the latest run ID
print(f"🔹 Latest Run ID: {latest_run_id}")

# 🔹 Register the model
model_uri = f"runs:/{latest_run_id}/model"
model_version = mlflow.register_model(model_uri, "BrainTumorSegmentationModel")
print(f"✅ Model successfully registered! Version: {model_version.version}")

# 🔎 Fetch the latest registered model version
latest_version = client.get_latest_versions("BrainTumorSegmentationModel")[0].version

# 🚀 Move model to Production
client.transition_model_version_stage(
    name="BrainTumorSegmentationModel",
    version=latest_version,  # Dynamically fetches the latest version
    stage="Production"
)

print(f"✅ Model Version {latest_version} moved to Production!")
