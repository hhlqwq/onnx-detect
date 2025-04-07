
import numpy as np
import onnxruntime
import torch
import cv2
from PIL import Image


def image_process(img_path,
                  shape=[640, 480],
                  means=[0.485, 0.456, 0.406],
                  stds=[0.229, 0.224, 0.225]):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img = np.transpose(img, (2, 0, 1))
    for t, mean, std in zip(img, means, stds):
        t = (t / 255 + mean) / std
    img = img.astype(np.float32)

    return img[np.newaxis, :, :, :]


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    'Generate anchors coordinates from scales/ratios'

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat([
        (boxes_ctr - anchors_ctr) / anchors_wh,
        torch.log(boxes_wh / anchors_wh)
    ], 1)


def delta2box(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

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


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
    'Box Decoding and Filtering'

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
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
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
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.size()[0]] = scores
        out_boxes[batch, :boxes.size()[0], :] = boxes
        out_classes[batch, :classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(bboxs, iou_threshold=0.5):
    return


def inference(input_data, model_path):

    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    detection = session.run([], {input_name: input_data})

    cls_heads = []
    box_heads = []
    for i in range(len(detection)):
        if i < 5:
            cls_heads.append(torch.from_numpy(detection[i]))
        else:
            box_heads.append(torch.from_numpy(detection[i]))

    # Inference post-processing
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        print(cls_head.shape)
        stride = input_data.shape[-1] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        decoded.append(decode(cls_head, box_head, stride,
                              threshold=0.35, top_n=10, anchors=anchors[stride]))

    # Perform non-maximum suppression
    decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
    # NMS threshold = 0.5
    scores, boxes, labels = decoded
    scores = scores.squeeze(0).numpy()
    boxes = boxes.squeeze(0).numpy()
    labels = labels.squeeze(0).numpy()

    bboxes = []
    for i in range(len(scores)):
        x = np.append(boxes[i], scores[i])
        x = np.append(x, int(labels[i]))
        bboxes.append[x]

    out = nms(bboxes)
    return out


def draw_detect(detections, img_path, out_path, detect_threshold=0.5):
    img = cv2.imread(img_path)
    bboxes = np.squeeze(detections[0])
    labels = detections[1].T
    scores = detections[2].T
    img_shape = img.shape

    len = np.sum(np.where(scores > detect_threshold, 1, 0))

    for i in range(len):
        x1 = img_shape[1] * bboxes[i][0]
        y1 = img_shape[0] * bboxes[i][1]
        x2 = img_shape[1] * bboxes[i][2]
        y2 = img_shape[0] * bboxes[i][3]

        thickness = max(round(min(img.shape) * 0.003), 2)
        font_thick = max(thickness - 1, 1)
        bbox_color = (255, 128, 0)

        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        message = "%s: %.2f" % (labels[i], scores[i])

        cv2.rectangle(img, p1, p2, bbox_color, thickness)
        cv2.putText(img, message, p1, cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 255), font_thick, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    img_path = "images/bus.jpeg"
    img = image_process(img_path)
    detection = inference(img, model_path="models/retinanet-9.onnx")
    draw_detect(detection, img_path, out_path="retinanet/bus_out.jpg")
