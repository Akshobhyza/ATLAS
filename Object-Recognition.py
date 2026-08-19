import cv2
from picamera2 import Picamera2
import time

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

MODEL = "MobileNetSSD_deploy.caffemodel"
CONFIG = "MobileNetSSD_deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(CONFIG, MODEL)

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Vision system started.")
print("Camera: 640x480")
print("Confidence threshold: 50%")
print("Press Ctrl+C to stop.")
print()

try:

    while True:

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        net.setInput(blob)
        detections = net.forward()

        best = None

        for i in range(detections.shape[2]):

            confidence = float(detections[0, 0, i, 2])

            if confidence < 0.50:
                continue

            class_id = int(detections[0, 0, i, 1])

            if class_id >= len(CLASSES):
                continue

            label = CLASSES[class_id]

            box = detections[0, 0, i, 3:7] * [
                frame.shape[1],
