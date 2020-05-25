import os
import cv2
import numpy
from imutils.video import FileVideoStream
from tqdm import tqdm
from skimage.measure import compare_ssim

actors = [f"Actor_{i}" for i in range(1, 25)]
emotions = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "suprised"}
emotion_count = dict(zip(list(emotions.keys()), [1] * 8))

print("[INFO] loading caffe model...")
caffe_net = cv2.dnn.readNetFromCaffe(
    "models/face_detection/deploy.prototxt.txt", "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")

print("[INFO] starting extraction process...")
for actor in tqdm(actors):

    videos = os.listdir(os.path.join("RAVDESS", actor))

    for video in tqdm(videos):

        fvs = FileVideoStream(os.path.join("RAVDESS", actor, video)).start()
        emotion = video.split("-")[2]
        prev_frame = None

        # loop over the frames from the video
        while True:

            try:
                frame = fvs.read()
                (h, w) = frame.shape[:2]
            except:
                break

            if prev_frame is not None:
                score, diff = compare_ssim(cv2.resize(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (335, 435)), cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (335, 435)), full=True)
                if score > 0.81:
                    continue

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and predictions
            caffe_net.setInput(blob)
            detections = caffe_net.forward()

            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence < 0.5:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.imwrite(os.path.join(
                    "datasets", "RAVDESS", emotions[emotion], emotions[emotion] + "." + str(emotion_count[emotion]) + ".jpg"),
                    frame[startY:endY, startX:endX])

                emotion_count[emotion] += 1

            prev_frame = frame

        fvs.stop()
