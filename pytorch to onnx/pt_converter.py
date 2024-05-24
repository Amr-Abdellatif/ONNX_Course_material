# Section 2: Conversion to ONNX
from pt_train import *
import torch.onnx

# Load the saved PyTorch model
model = SimpleNN()
model.load_state_dict(torch.load(torch_model_path))
model.eval()

# Define the model path
onnx_model_path = './output/iris_model.onnx'

# Convert the PyTorch model to ONNX
dummy_input = torch.tensor(X_train[0:1], dtype=torch.float32)
torch.onnx.export(model, dummy_input, onnx_model_path, 
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=13)

print(f'ONNX model saved to {onnx_model_path}')
