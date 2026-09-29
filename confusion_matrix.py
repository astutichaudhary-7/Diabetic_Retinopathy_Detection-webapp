import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFConvNextModel

IMG_SIZE = 224
NUM_CLASSES = 5
CONVNEXT_CHECKPOINT = "facebook/convnext-tiny-224"
CONVNEXT_MODEL_PATH = "model/dr_convnext_model.weights.h5"

# Must match training normalization exactly
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]


def build_convnext_model():
    convnext_backbone = TFConvNextModel.from_pretrained(CONVNEXT_CHECKPOINT)
    convnext_backbone.trainable = False

    pixel_input = Input(shape=(3, IMG_SIZE, IMG_SIZE), name="pixel_values")
    convnext_output = convnext_backbone(pixel_values=pixel_input).pooler_output

    x = Dropout(0.3)(convnext_output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=pixel_input, outputs=output)


model = build_convnext_model()
model.load_weights(CONVNEXT_MODEL_PATH)


def predict_grade(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = img / 255.0

    img = (img - IMAGENET_MEAN) / IMAGENET_STD   # same normalization used during training

    img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return classes[index], confidence