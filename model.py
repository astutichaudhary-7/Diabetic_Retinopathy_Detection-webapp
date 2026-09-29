import pickle
import cv2
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


# ---------------------------------------------------------------------------
# Grad-CAM setup
# ---------------------------------------------------------------------------
def _find_last_conv_layer(model):
    """
    Finds the last 4D-output (conv feature map) layer, i.e. the layer
    right before EfficientNetV2S's global average pooling.
    For EfficientNetV2S this is normally "top_activation". If your model
    prints an error here, run model.summary(), find the last conv/activation
    layer name yourself, and hardcode it below instead of calling this.
    """
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError(
        "Couldn't auto-detect a conv layer for Grad-CAM. Run model.summary() "
        "and set LAST_CONV_LAYER manually below."
    )


LAST_CONV_LAYER = _find_last_conv_layer(model)
# LAST_CONV_LAYER = "top_activation"  # <-- uncomment + edit if auto-detect fails

_grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
)


def make_gradcam_heatmap(img_array, pred_index=None):
    """
    img_array: preprocessed batch of shape (1, IMG_SIZE, IMG_SIZE, 3) —
    same array you'd pass to model.predict().
    Returns: heatmap (2D numpy array, values 0-1), predicted class index,
    full softmax probabilities.
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = _grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def save_gradcam_overlay(image_path, save_path, img_array, pred_index, alpha=0.4):
    """
    Draws the Grad-CAM heatmap on top of the original image and saves it
    to save_path (e.g. static/gradcam/<filename>).
    """
    heatmap, _, _ = make_gradcam_heatmap(img_array, pred_index)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlaid = cv2.addWeighted(heatmap_color, alpha, original, 1 - alpha, 0)
    cv2.imwrite(save_path, overlaid)


def predict_grade(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # returned so app.py can generate the Grad-CAM overlay without
    # re-loading and re-predicting the same image a second time
    return classes[index], confidence, img, int(index)