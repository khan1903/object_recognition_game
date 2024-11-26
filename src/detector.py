import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # Load YOLO
        self.net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getLayers() if i[0] in self.net.getUnconnectedOutLayers()]
        self.classes = []
        with open('data/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_objects(self, frame):
        # Prepare the frame for YOLO detection
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Analyze detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []

        if len(indexes) > 0:
            for i in indexes.flatten():
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                bbox = boxes[i]
                detections.append((label, confidence, bbox))

        return detections
