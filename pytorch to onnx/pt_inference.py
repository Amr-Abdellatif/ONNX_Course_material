# Section 3: Inference

import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = './output/iris_model.onnx'

onnx_model = ort.InferenceSession(onnx_model_path)

# Provided data for inference
data = np.array([[4.5, 4.9, 5.1, 5.4],
                 [1.5, 2.9, 3.1, 1.4],
                 [7.5, 6.9, 8.1, 6.4]], dtype=np.float32)

def prepare_input(data):
    return {onnx_model.get_inputs()[0].name: data}

# Make predictions
input_data = prepare_input(data)
predictions = onnx_model.run(None, input_data)

# Convert the predictions to class labels
predicted_labels = np.argmax(predictions[0], axis=1)
print(f'Predicted labels: {predicted_labels}')
