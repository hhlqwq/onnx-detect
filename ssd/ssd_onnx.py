# from os.path import
import cv2
import numpy as np
import onnx
import torch
import itertools
import torch.nn.functional as F
import onnxruntime
from math import sqrt


class SSDCase():
    def __init__(self, class_num=80) -> None:
        self.nb = 4
        self.no = class_num
        self.input_size = [1200, 1200]
        self.aspect_radios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.scale_xy = 0.1
        self.scale_wh = 0.2
        self.scale = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]

        self.colors = [
            (255, 50, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (207, 210, 49),
            (72, 249, 10),
            (146, 204, 23),
            (61, 219, 134),
            (26, 147, 52),
            (0, 212, 187),
            (44, 153, 168),
            (0, 194, 255),
            (52, 69, 147),
            (100, 115, 255),
            (0, 24, 236),
            (132, 56, 255),
            (82, 0, 133),
            (203, 56, 255),
            (255, 149, 200),
            (255, 55, 199),
        ]

    def _get_default_boxes(self, feat_size):
        steps = [(int(self.input_size[0] / fs[0]),
                  int(self.input_size[1] / fs[1])) for fs in feat_size]
        scales = [(int(s * self.input_size[0]), int(s * self.input_size[1]))
                  for s in self.scale]
        steps_w = [s[0] for s in steps]
        steps_h = [s[1] for s in steps]
        fkw = self.input_size[0] // np.array(steps_w)
        fkh = self.input_size[1] // np.array(steps_h)

        default_boxes = []
        for idx, feature in enumerate(feat_size):
            feat_w, feat_h = feature
            sk1 = scales[idx][0] / self.input_size[0]
            sk2 = scales[idx + 1][1] / self.input_size[1]
            sk3 = sqrt(sk1 * sk2)
            all_size = [(sk1, sk1), (sk3, sk3)]
            for alpha in self.aspect_radios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_size.append((w, h))
                all_size.append((h, w))
            for w, h in all_size:
                for i, j in itertools.product(range(feat_w, range(feat_h))):
                    x, y = (j + 0.5) / fkh[idx], (i + 0.5) / fkw[idx]
                    default_boxes.append(x,y,w,h)
        dboxes = torch.tensor(default_boxes)
        dboxes.clamp_(min=0, max=1)

        return dboxes

    def _scale_back_batch(self, boxes, scores, feat_size):
        

"/Users/he/coco/val2017"
