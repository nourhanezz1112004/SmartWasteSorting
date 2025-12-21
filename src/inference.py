"""Inference utilities: load model, preprocess image, predict."""
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

_model = None
_class_indices = None

def load(wpath, class_indices=None):
    global _model, _class_indices
    _model = load_model(wpath)
    _class_indices = class_indices

def preprocess_image(frame, target_size=(224,224)):
    # frame is a BGR OpenCV image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict(frame):
    if _model is None:
        raise RuntimeError('Model not loaded. Call load first.')
    x = preprocess_image(frame)
    probs = _model.predict(x)[0]
    if _class_indices:
        labels = [k for k, v in sorted(_class_indices.items(), key=lambda x: x[1])]

    else:
        # numeric labels
        labels = [str(i) for i in range(len(probs))]
    idx = probs.argmax()
    return labels[idx], float(probs[idx])
