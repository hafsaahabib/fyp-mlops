import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model

# Set MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Load the locally saved .h5 model using TensorFlow
model_path = "3D-UNet-final-model.h5"
model = load_model(model_path)  # ✅ Load it properly first

# Start an MLflow run and log the model in MLflow format
with mlflow.start_run():
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 32)

    # Correctly log the model to MLflow so it can be used later
    mlflow.keras.log_model(model, "model")

print("✅ Model successfully logged to MLflow!")
