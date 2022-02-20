import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import io

import json
import base64
import requests

import tensorflow as tf
import numpy as np
import cv2
import imutils
from imutils.contours import sort_contours
from PIL import Image

IMAGE_SIZE = (700, 700)

MODEL_URI_NUMBERS = 'https://numbers-model-container.herokuapp.com/v1/models/numbers_model:predict'
MODEL_URI_CHARACHTERS = 'https://letter-model-container.herokuapp.com/v1/models/letters_model:predict'

LABELS_NUMBERS = "0123456789"
LABELS_NUMBERS = [l for l in LABELS_NUMBERS]

LABELS_CHARACHTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABELS_CHARACHTERS = [l for l in LABELS_CHARACHTERS]

def get_prediction(min_width, max_width, min_height, max_height, model_predict, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (700,700))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if (w >= int(min_width) and w <= int(max_width)) and (h >= int(min_height) and h <= int(max_height)):

            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=28)

            else:
                thresh = imutils.resize(thresh, height=28)

            (tH, tW) = thresh.shape
            dX = int(max(0, 28 - tW) / 2.0)
            dY = int(max(0, 28 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))

            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # Creating body for TensorFlow serving request
    data = json.dumps({"signature_name": "serving_default", "instances": chars.tolist()})
    print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

    headers = {"content-type": "application/json"}

    # Making POST request
    if (model_predict == 'numbers'):
        r = requests.post(MODEL_URI_NUMBERS, data=data, headers=headers)
    else:
        r = requests.post(MODEL_URI_CHARACHTERS, data=data, headers=headers)
    
    r.raise_for_status()
    # Decoding results from TensorFlow Serving server
    preds = json.loads(r.content.decode('utf-8'))

    prediction_list = []

    for (pred, (x, y, w, h)) in zip(preds['predictions'], boxes):
        i = np.argmax(pred)
        prob = pred[i]
        if (model_predict == 'numbers'):
            label = LABELS_NUMBERS[i]
        else:
            label = LABELS_CHARACHTERS[i]

        item = {"character": label, "prediction": round(prob * 100, 2)}
        prediction_list.append(item)

        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    image.save(rawBytes, 'png')
    rawBytes.seek(0)
    image = base64.b64encode(rawBytes.read()).decode('ascii')
    data = {
        "prediction_list": prediction_list,
        "image": image
    }
    
    return data

