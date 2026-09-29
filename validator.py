import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224
GATEKEEPER_MODEL_PATH = "model/gatekeeper_model.h5"
THRESHOLD = 0.5

_gatekeeper_model = None


def _build_gatekeeper_architecture():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=output)


def _load_gatekeeper():
    global _gatekeeper_model
    if _gatekeeper_model is None:
        _gatekeeper_model = _build_gatekeeper_architecture()
        _gatekeeper_model.load_weights(GATEKEEPER_MODEL_PATH)
    return _gatekeeper_model


def is_retina_image(filepath, threshold=THRESHOLD):
    model = _load_gatekeeper()

    img = load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    prob_retina = float(model.predict(arr, verbose=0)[0][0])

    is_retina = prob_retina >= threshold
    confidence = prob_retina if is_retina else (1 - prob_retina)

    return is_retina, confidence