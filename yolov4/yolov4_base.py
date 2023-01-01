import cv2
import numpy as np
import onnx
from scipy import special
import onnxruntime

def load_input(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = img[np.newaxis, ...].astype(np.float32)   # 416,416,3->1,416,416,3
    return img

    