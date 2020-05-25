import cv2
import numpy as np
import tensorflow as tf

from collections import deque, OrderedDict
from tensorflow.keras.models import load_model


class EmotionRecognition():

    EMOTIONS = ["angry", "disgust", "fear",
                "happy", "sad", "suprised", "neutral"]
    EMOTION_COLORS = [(0, 0, 255), (0, 255, 0), (0, 0, 0),
                      (255, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]

    CAFFE_NET = cv2.dnn.readNetFromCaffe("../models/face_detection/deploy.prototxt.txt",
                                         "../models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
    EMOTION_CLASSIFIER = load_model(
        "../models/trained_fer_models/mini_xception.0.65-119.hdf5")

    def __init__(self, deque_size=4):
        self.deque_size = deque_size
        self.faces = OrderedDict()

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.CAFFE_NET.setInput(blob)
        detections = self.CAFFE_NET.forward()

        rects = list()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            rects.append(box.astype("int"))

        self.rects = rects
        self.frame = frame
        return rects

    def detect_emotions(self, objects):
        detected_emotions_on_faces = OrderedDict()
        for (object_id, centroid) in objects.items():

            for rect in self.rects:
                (start_x, start_y, end_x, end_y) = rect
                c_x = int((start_x + end_x) / 2.0)
                c_y = int((start_y + end_y) / 2.0)

                if np.array_equal([c_x, c_y], centroid):
                    face = self.frame[start_y:end_y, start_x:end_x]
                    try:
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    except:
                        break

                    gray_face = cv2.resize(
                        gray_face, (self.EMOTION_CLASSIFIER.input_shape[1:3]))
                    gray_face = gray_face.astype("float32")
                    gray_face /= 255.0
                    gray_face -= 0.5
                    gray_face *= 2.0
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)

                    emotion_prediction = self.EMOTION_CLASSIFIER.predict(gray_face)[
                        0]

                    if object_id not in self.faces:
                        self.faces[object_id] = deque(maxlen=self.deque_size)

                    self.faces[object_id].append(emotion_prediction)
                    results = np.array(self.faces[object_id]).mean(axis=0)
                    emotion = self.EMOTIONS[np.argmax(results)]

                    detected_emotions_on_faces[object_id] = {"id": object_id,
                                                             "emotion": emotion}

                    text = f"{emotion}"
                    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
                    cv2.rectangle(self.frame, (start_x, start_y),
                                  (end_x, end_y), self.EMOTION_COLORS[np.argmax(results)], 1)
                    cv2.putText(self.frame, text, (start_x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.EMOTION_COLORS[np.argmax(results)], 1)

                    text_id = f"ID {object_id}"
                    cv2.putText(self.frame, text_id, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                    cv2.circle(
                        self.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                    break

        return self.frame, detected_emotions_on_faces
