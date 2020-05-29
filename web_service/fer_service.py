import cv2
import base64
import gc
import sqlite3
import numpy as np
import tensorflow as tf

from flask_cors import CORS
from flask import Flask, jsonify, request

from collections import OrderedDict
from centroidtracker import CentroidTracker
from emotionrecognition import EmotionRecognition


app = Flask(__name__)

CORS(app)

ct = CentroidTracker(max_dissapeared=10)
er = EmotionRecognition()


@app.route("/api/emotion-detection", methods=["POST"])
def detect_emotion_in_frame():
    global ct, er
    detected_emotions_on_faces = OrderedDict()

    request_data = request.get_json()

    if request_data["clearGlobals"]:
        print("reset signal sent")
        ct = CentroidTracker(max_dissapeared=10)
        er = EmotionRecognition()
        gc.collect()

    try:
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(
            request_data["capturedFrame"][23:]), np.uint8), cv2.IMREAD_COLOR)
    except:
        return jsonify({"error": "error"})

    rects = er.detect_faces(frame)
    objects, dissapeared = ct.track(rects)
    frame, detected_emotions_on_faces = er.detect_emotions(objects)

    flag, encoded_img = cv2.imencode(".jpg", frame)

    return jsonify({"frame": base64.b64encode(encoded_img).decode("utf-8"),
                    "detectedEmotionsOnFaces": list(detected_emotions_on_faces.values()),
                    "dissapearedFaces": list({"id": d[0], "counter": d[1]} for d in dissapeared.items())})


@app.route("/api/save-session-data", methods=["POST"])
def save_session_data():

    request_data = request.get_json()

    print(request_data)

    try:
        conn = sqlite3.connect("database/fer_db.db")
        conn.execute("INSERT INTO SESSION_RESULTS(SESSION_START_TIME, DURATION, MAX_DETECTED_FACE, MOST_DETECTED_EMOTION, MOST_DETECTED_EMOTION_COUNTER) VALUES(?, ?, ?, ?, ?)",
                        (request_data["sessionStartTimestamp"], request_data["duration"], request_data["maxDetectedFace"], request_data["mostDetecedEmotion"]["emotion"], request_data["mostDetecedEmotion"]["counter"]))
        conn.commit()
        conn.close()
        return jsonify({"success": "Session saved successfully"})
    except Exception as e:
        print(e)
        conn.close()
        return jsonify({"error": "Session cannot saved"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)
