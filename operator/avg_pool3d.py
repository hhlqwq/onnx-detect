import numpy as np
from onnx import TensorProto,numpy_helper
from onnx.helper import (
    make_node, make_tensor_value_info, make_graph, make_model
)
# import easydict
import onnxruntime
import numpy as np

print("Supported onnxruntime version: ", onnxruntime.__version__)
print("Supported Opset versions: ", onnxruntime.get_available_providers())

class AvgPool3d():        
    
    def __init__(self,):                
        pass
        
    def inference_onnx(self,inputs,args):
        
        x = make_tensor_value_info(name="x",elem_type=TensorProto.FLOAT, shape=[1,8,32,32,32])
        y = make_tensor_value_info(name="y",elem_type=TensorProto.FLOAT, shape=[1,8,16,16,16])

        op = make_node(op_type="AveragePool",name="avg_pool3d",inputs=["x"],outputs=["y"],kernel_shape=[2,2,2],strides=[2,2,2])
        g = make_graph(nodes=[op],name="test_pool3d",inputs=[x],outputs=[y])
        m = make_model(graph=g)

        sess = onnxruntime.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
        res = sess.run(None, {'x': inputs})
        
        return res[0]

    def inference_numpy(self,inputs,args):
        N,C,D,H,W = inputs.shape
        res = np.zeros([N,C,D//2,H//2,W//2])
        
        for i in range(0,D,2):
            for j in range(0,H,2):
                for k in range(0,W,2):
                    tmp = inputs[:,:,i:i+2,j:j+2,k:k+2]
                    value = tmp.mean((2,3,4))
                    
                    res[:,:,i//2,j//2,k//2] = value
                    
        return res
    
    def compare(self,):
        x = np.arange(1*8*32*32*32).reshape(1,8,32,32,32).astype(np.float32)
        # x = np.random.rand(1,8,32,32,32).astype(np.float32)

        res_np=self.inference_numpy(x,None)
        res_onnx = self.inference_onnx(x,None)

        print(res_np.shape,res_onnx.shape)
        print(np.abs(res_np-res_onnx).mean())

t = AvgPool3d()
t.compare()
