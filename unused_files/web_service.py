# %%
import cv2
import base64
import threading
import numpy as np
import tensorflow as tf

from flask_cors import CORS
from flask import Flask, jsonify, request

from collections import deque, OrderedDict
from centroidtracker import CentroidTracker
from tensorflow.keras.models import load_model

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %%
DEQUE_SIZE = 4
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "suprised", "neutral"]
EMOTION_COLORS = [(0, 0, 255), (0, 255, 0), (0, 0, 0),
                  (255, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]


CAFFE_NET = cv2.dnn.readNetFromCaffe("models/face_detection/deploy.prototxt.txt",
                                     "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
EMOTION_CLASSIFIER = load_model(
    "models/trained_fer_models/mini_xception.0.65-119.hdf5")

# %%
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

faces = OrderedDict()
ct = CentroidTracker(max_dissapeared=10)

lock = threading.Lock()


@app.route("/api/emotion-detection", methods=["POST"])
def detect_emotion_in_frame():
    global faces, ct, lock
    detected_emotions_on_faces = OrderedDict()
    with lock:
        request_data = request.get_json()
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(
            request_data["capturedFrame"][23:]), np.uint8), cv2.IMREAD_COLOR)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        CAFFE_NET.setInput(blob)
        detections = CAFFE_NET.forward()

        rects = list()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.6:
                continue

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

                    if object_id not in faces:
                        faces[object_id] = deque(maxlen=DEQUE_SIZE)

                    faces[object_id].append(emotion_prediction)
                    results = np.array(faces[object_id]).mean(axis=0)
                    emotion = EMOTIONS[np.argmax(results)]

                    detected_emotions_on_faces[object_id] = {"id": object_id,
                                                             "emotion": emotion}

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

    flag, encoded_img = cv2.imencode(".jpg", frame)

    return jsonify({"flag": flag,
                    "frame": base64.b64encode(encoded_img).decode("utf-8"),
                    "detectedEmotionsOnFaces": list(detected_emotions_on_faces.values())})


if __name__ == "__main__":

    t = threading.Thread(target=detect_emotion_in_frame)
    t.daemon = True
    t.start()

    app.run(host="localhost", debug=True, threaded=True)
