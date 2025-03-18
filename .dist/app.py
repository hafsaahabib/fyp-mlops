import mlflow.pyfunc
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the latest registered model from MLflow
model = mlflow.pyfunc.load_model("models:/BrainTumorSegmentationModel/Production")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = np.array(request.json["image"])  # Convert input MRI scan to NumPy array
    output = model.predict(input_data)  # Run inference
    return jsonify({"prediction": output.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
