import cv2
import base64
import gc
import tracemalloc
import numpy as np
import tensorflow as tf

from flask_cors import cross_origin
from flask import Flask, jsonify, request

from collections import OrderedDict
from centroidtracker import CentroidTracker
from emotionrecognition import EmotionRecognition


app = Flask(__name__)

ct = CentroidTracker(max_dissapeared=10)
er = EmotionRecognition()

tracemalloc.start()
s = None


@app.route("/api/emotion-detection", methods=["POST"])
@cross_origin()
def detect_emotion_in_frame():
    global ct, er
    detected_emotions_on_faces = OrderedDict()

    request_data = request.get_json()

    if request_data["clearGlobals"]:
        print("reset signal sent")
        ct = CentroidTracker(max_dissapeared=10)
        er = EmotionRecognition()

    try:
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(
            request_data["capturedFrame"][23:]), np.uint8), cv2.IMREAD_COLOR)
    except:
        return jsonify({"error": "error"})

    rects = er.detect_faces(frame)
    objects, dissapeared = ct.track(rects)
    frame, detected_emotions_on_faces = er.detect_emotions(objects)

    flag, encoded_img = cv2.imencode(".jpg", frame)

    gc.collect()
    return jsonify({"frame": base64.b64encode(encoded_img).decode("utf-8"),
                    "detectedEmotionsOnFaces": list(detected_emotions_on_faces.values()),
                    "dissapearedFaces": list({"id": d[0], "counter": d[1]} for d in dissapeared.items())})


@app.route("/snapshot", methods=["GET"])
def snap():
    global s
    if not s:
        s = tracemalloc.take_snapshot()
        return "snapshot taken\n"
    else:
        lines = []
        top_stats = tracemalloc.take_snapshot().compare_to(s, "lineno")
        for stat in top_stats[:10]:
            lines.append(str(stat))
        return "\n".join(lines)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)
