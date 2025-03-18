import requests
import json
import numpy as np

url = "http://127.0.0.1:5001/invocations"

# Generate a random input matching your model's expected shape (128x128x4)
input_data = np.random.randint(0, 256, (1, 128, 128, 4)).tolist()

data = {"instances": input_data}

headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)

# Print only the first few predictions for readability
output = response.json()
print("Response (First 2 Predictions):", output["predictions"][:2])
