# Section 2: Conversion to ONNX

import tf2onnx
import onnx
from tf_train import *
# Define the model path
onnx_model_path = './output/iris_model.onnx'

# Convert the TensorFlow model to ONNX
spec = (tf.TensorSpec((None, X_train.shape[1]), tf.float32, name="input"),)
output_path = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_model_path)

print(f'Model saved to {onnx_model_path}')
