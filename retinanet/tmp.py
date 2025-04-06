import os
import sys
from os.path import dirname, join, splitext
import cv2
import numpy as np
import torch
import onnx
from scipy import special
import onnxruntime


class RETENANETTestCase():
    def __init__(self, class_num=1):
        self.ratio = [1.0, 2.0, 0.5]
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]

        self.nc = class_num
        self.no = self.nc + 5
        self.colors = [
            (255, 56, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (287, 210, 49),
            (72, 249, 10),
            (146, 204, 23),
            (61, 219, 134),
            (26, 147, 52),
            (0, 212, 187),
            (44, 153, 168),
            (0, 194, 255),
            (52, 69, 147),
            (190, 115, 255),
            (6, 24, 236),
            (132, 56, 255),
            (82, 0, 133),
            (203, 56, 255),
            (255, 149, 200),
            (255, 55, 199),
        ]

    def _draw_path(self):
        return g_output_img_path

    def _delta2box(self, deltas, anchors, size, stride):
        anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
        ctr = anchors[:, :2] + 0.5 * anchors_wh
        pred_ctr = deltas[:, :2] * anchors_wh + ctr
        pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

        m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
        M = (torch.tensor([size], device=deltas.device,
                          dtype=deltas.dtype) * stride - 1)

        def clamp(t): return torch.max(m, torch.min(t, M))
        return torch.cat([
            clamp(pred_ctr - 0.5 * pred_wh),
            clamp(pred_ctr + 0.5 * pred_wh - 1)
        ], 1)

    def _decode(self, all_cls_head, all_box_head, stride, threshold=0.35,
                top_n=1000, anchors=None, rotated=False):
        if rotated:
            anchors = anchors[0]
        num_boxes = 4 if not rotated else 6

        device = "cpu"
        anchors = anchors.to(device).type(all_cls_head.type())
        num_anchors = anchors.size()[0] if anchors is not None else 1
        num_classes = all_cls_head.size()[1] // num_anchors
        height, width = all_cls_head.size()[-2:]

        batch_size = all_cls_head.size()[0]
        out_scores = torch.zeros((batch_size, top_n), device=device)
        out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
        out_classes = torch.zeros((batch_size, top_n), device=device)

        # Per item in batch
        for batch in range(batch_size):
            cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
            box_head = all_box_head[batch, :, :,
                                    :].contiguous().view(-1, num_boxes)

            # Keep scores over threshold
            keep = (cls_head >= threshold).nonzero().view(-1)
            if keep.nelement() == 0:
                continue

            # Gather top elements
            scores = torch.index_select(cls_head, 0, keep)
            scores, indices = torch.topk(
                scores, min(top_n, keep.size()[0]), dim=0)
            indices = torch.index_select(keep, 0, indices).view(-1)
            classes = (indices / width / height) % num_classes
            classes = classes.type(all_cls_head.type())

            # Infer kept bboxes
            x = indices % width
            y = (indices / width).type(torch.LongTensor) % height
            a = (((indices / num_classes).type(torch.LongTensor) /
                  height).type(torch.LongTensor) / width).type(torch.LongTensor)
            box_head = box_head.view(num_anchors, num_boxes, height, width)
            boxes = box_head[a, :, y, x]

            if anchors is not None:
                grid = torch.stack([x, y, x, y], 1).type(
                    all_cls_head.type()) * stride + anchors[a, :]
                boxes = self._delta2box(boxes, grid, [width, height], stride)

            out_scores[batch, :scores.size()[0]] = scores
            out_boxes[batch, :boxes.size()[0], :] = boxes
            out_classes[batch, :classes.size()[0]] = classes

        return out_scores, out_boxes, out_classes

    def _generate_anchors(self, stride):
        scales = torch.FloatTensor(self.scales).repeat(len(self.ratio), 1)
        scales = scales.transpose(0, 1).contiguous().view(-1, 1)
        ratios = torch.FloatTensor(self.ratio * len(self.scales))

        wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
        ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
        dwh = torch.stack([ws, ws * ratios], dim=1)
        xy1 = 0.5 * (wh - dwh * scales)
        xy2 = 0.5 * (wh + dwh * scales)

        return torch.cat([xy1, xy2], dim=1)

    def _nms(self, bboxes, scores, iou_threshold=0.35):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        res = []
        index = scores.argsort()[::-1]
        while index.size > 0:
            i = index[0]
            res.append(i)
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x1[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11)
            h = np.maximum(0, y22 - y11)

            overlaps = w * h
            np.seterr(divide='ignore', invalid='ignore')
            iou = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(iou <= iou_threshold)[0]
            index = index[idx + 1]
        return np.array(res)

    def _postprocess(self, cls_heads, box_heads, max_det=1000, threshold=0.35):
        anchors = {}
        decoded = []

        for cls_head, box_head in zip(cls_heads, box_heads):
            stride = 640 // cls_head.shape[-1]
            if stride not in anchors:
                anchors[stride] = self._generate_anchors(stride)

            decoded.append(self._decode(cls_head, box_head, stride,
                                        threshold, top_n=100,
                                        anchors=anchors[stride]))
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]

        out_scores, out_boxes, out_classes = decoded

        out_scores = out_scores.squeeze(0).numpy()
        out_boxes = out_boxes.squeeze(0).numpy()
        out_classes = out_classes.squeeze(0).numpy()

        i = self._nms(out_boxes, out_scores)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        out = []
        for index in i:
            if (out_scores[index] > threshold):
                x = np.append(out_boxes[index, :], out_scores[index])
                x = np.append(x, int(out_classes[index]))
                out.append(x)

        return out

    def _sigmoid(self, x):
        return .5 * (1 + np.tanh(.5 * x))

    def inference_module(self, detections):
        x = detections

        cls_heads = []
        box_heads = []
        for i in range(len(x)):
            if i < 5:
                x[i] = self._sigmoid(x[i])
                cls_heads.append(torch.from_numpy(x[i]))
            else:
                box_heads.append(torch.from_numpy(x[i]))

        prediction = self._postprocess(cls_heads, box_heads)

        return prediction

    def check_top_n(self, detections, shape=[480, 640]):
        input_path = g_input_img_path

        suffix = splitext(input_path)[-1]
        if suffix not in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
            print("input path not support. path:", input_path)
            return

        height, width = shape
        im = cv2.imread(input_path)
        im = cv2.resize(im, (width, height))

        thickness = max(round(min(im.shape) * 0.003), 2)
        txt_color = (255, 255, 255)

        bboxes = detections
        for box in bboxes:
            color = self.colors[int(box[5]) % len(self.colors)]
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            im = cv2.rectangle(im, p1, p2, color, thickness)
            lable = f"{box[4]:.2f}"

            scale = thickness / 3
            font_thickness = max(thickness - 1, 1)
            w, h = cv2.getTextSize(lable, 0, scale, font_thickness)[0]
            outside = p1[1] - h - 3 >= 0
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(
                im,
                lable,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                scale,
                txt_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        draw_path = self._draw_path()
        cv2.imwrite(draw_path, im)
        try:
            os.chmod(draw_path, 0o777)
        except:
            pass


def load_input(img_path,
               mean=np.array([0.485, 0.456, 0.406]),
               std=np.array([0.229, 0.224, 0.2251])):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis, ...].astype(np.float32)
    img = np.transpose(img, [0, 3, 1, 2])

    for i in range(img. shape[1]):
        img[:, i, :, :] = (img[:, i, :, :] / 255 - mean[i]) / std[i]

    return img


def load_model(model_filename, output_names):
    model = onnx.load(model_filename)

    for output_name in output_names:
        output_node = onnx.ValueInfoProto()
        output_node.name = output_name

        model.graph.output.append(output_node)

    return onnxruntime.InferenceSession(model.SerializeToString())


def inference(model, input_data, output_names):
    return model.run(
        output_names, {"input": input_data}
    )


g_input_img_path = "images/bus.jpeg"
g_output_img_path = "retinanet/bus_out.jpg"
g_model_path = "models/retinanet-9.onnx"

if __name__ == "__main__":
    output_names = ["1042", "1051", "1060", "1069", "1078",
                    "output6", "output7", "output8", "output9", "output10"]
    model = load_model(g_model_path, output_names)
    input_data = load_input(g_input_img_path)
    detections = inference(model, input_data, output_names)

    testcase = RETENANETTestCase(class_num=80)
    prediction = testcase.inference_module(detections)

    testcase.check_top_n(prediction)
