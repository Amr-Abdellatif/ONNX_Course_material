import onnxruntime as rt
import numpy as np

data = np.array([[4.5,4.9,5.1,5.4],[1.5,2.9,3.1,1.4],[7.5,6.9,8.1,6.4]])

sess = rt.InferenceSession("./output/model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onnx = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
print(pred_onnx)