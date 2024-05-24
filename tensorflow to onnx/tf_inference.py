# Section 3: Inference

import onnxruntime as ort
import numpy as np
from tf_train import X_test,y_test

# Load the ONNX model
onnx_model_path = './output/iris_model.onnx'
onnx_model = ort.InferenceSession(onnx_model_path)

# Prepare the input data for inference
def prepare_input(data):
    return {onnx_model.get_inputs()[0].name: data.astype(np.float32)}

# Make predictions
input_data = prepare_input(X_test)
predictions = onnx_model.run(None, input_data)

# Convert the predictions to class labels
predicted_labels = np.argmax(predictions[0], axis=1)
true_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f'Inference Accuracy: {accuracy:.4f}')
