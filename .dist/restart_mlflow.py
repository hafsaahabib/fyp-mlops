import os

# Stop the previous MLflow server process (if running)
os.system("taskkill /IM mlflow.exe /F")

# Start the MLflow server with the new model
os.system('start cmd /k "mlflow models serve -m models:/BrainTumorSegmentationModel/Production --port 5001 --no-conda"')

print("✅ MLflow model server restarted with the latest model!")
