import cv2
import numpy as np
import onnx
import onnxruntime


def image_process(img_path,
                  target_size=[1200, 1200],
                  mean=np.array([0.485, 0.456, 0.406]),
                  std=np.array([0.229, 0.224, 0.225])):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    img = np.array(img).astype(np.float32)
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)

    for i in range(img.shape[1]):
        img[:, i, :, :] = (img[:, i, :, :] / 255 - mean[i]) / std[i]

    return img


def inference(input_data, model_path):
    sess = onnxruntime.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    detection = sess.run([], {input_name: input_data})

    return detection


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
    input = image_process(img_path)
    detection = inference(input, model_path="models/ssd-10.onnx")
    draw_detect(detection, img_path, out_path="ssd/bus_out.jpg")
