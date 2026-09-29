import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224
NUM_CLASSES = 5
FULL_WEIGHTS_PATH = "model/dr_efficientnet_weights.pkl"

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]


def build_efficientnet_model():
    base = EfficientNetV2S(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    x = Dropout(0.3)(base.output)
    x = Dense(128, activation="relu", name="head_dense_128")(x)
    x = Dropout(0.2)(x)
    output = Dense(NUM_CLASSES, activation="softmax", name="head_dense_output")(x)
    return Model(inputs=base.input, outputs=output)


model = build_efficientnet_model()

with open(FULL_WEIGHTS_PATH, "rb") as f:
    all_weights = pickle.load(f)

model.set_weights(all_weights)

image_path = "input.png"

img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
index = np.argmax(prediction)

print("Prediction :", classes[index])