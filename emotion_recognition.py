import os
import cv2
import time
import base64
import numpy as np

from collections import deque, OrderedDict
from imutils.video import FileVideoStream, VideoStream
from centroidtracker import CentroidTracker
from tensorflow.keras.models import load_model


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "suprised", "neutral"]
EMOTION_COLORS = [(0, 0, 255), (0, 255, 0), (0, 0, 0),
                  (255, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]


CAFFE_NET = cv2.dnn.readNetFromCaffe("models/face_detection/deploy.prototxt.txt",
                                     "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
EMOTION_CLASSIFIER = load_model(
    "models/trained_fer_models/mini_xception.0.65-119.hdf5")

for test_video in os.listdir("test_videos"):

    faces = OrderedDict()
    ct = CentroidTracker()
    vs = FileVideoStream("test_videos/" + test_video).start()
    # vs = VideoStream(0).start()
    # time.sleep(2)

    while True:

        try:
            frame = vs.read()
            (h, w) = frame.shape[:2]
        except:
            break

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        CAFFE_NET.setInput(blob)
        detections = CAFFE_NET.forward()

        rects = list()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence < 0.6:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            rects.append(box.astype("int"))

        objects = ct.track(rects)

        for (object_id, centroid) in objects.items():

            for rect in rects:
                (start_x, start_y, end_x, end_y) = rect
                c_x = int((start_x + end_x) / 2.0)
                c_y = int((start_y + end_y) / 2.0)

                if np.array_equal([c_x, c_y], centroid):
                    face = frame[start_y:end_y, start_x:end_x]
                    try:
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    except:
                        break

                    gray_face = cv2.resize(
                        gray_face, (EMOTION_CLASSIFIER.input_shape[1:3]))
                    gray_face = gray_face.astype("float32")
                    gray_face /= 255.0
                    gray_face -= 0.5
                    gray_face *= 2.0
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)

                    emotion_prediction = EMOTION_CLASSIFIER.predict(gray_face)[
                        0]
                    # emotion_probability = np.max(emotion_prediction)
                    # emotion = EMOTIONS[np.argmax(emotion_prediction)]

                    if object_id not in faces:
                        faces[object_id] = deque(maxlen=16)

                    faces[object_id].append(emotion_prediction)
                    results = np.array(faces[object_id]).mean(axis=0)
                    emotion = EMOTIONS[np.argmax(results)]

                    # draw the bounding box of the face along with the associated probability
                    text = f"{emotion}"
                    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
                    cv2.rectangle(frame, (start_x, start_y),
                                  (end_x, end_y), EMOTION_COLORS[np.argmax(results)], 1)
                    cv2.putText(frame, text, (start_x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, EMOTION_COLORS[np.argmax(results)], 1)

                    text_id = f"ID {object_id}"
                    cv2.putText(frame, text_id, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.circle(
                        frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    break

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()
    break
