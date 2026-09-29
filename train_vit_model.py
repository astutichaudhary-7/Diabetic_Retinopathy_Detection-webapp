"""
train_vit_model.py

Trains the main 5-class Diabetic Retinopathy classifier using a
Vision Transformer (ViT-B16, pretrained on ImageNet21k) instead of a
plain CNN, via transfer learning.

Classes: No_DR, Mild, Moderate, Severe, Proliferate_DR

Run:
    python train_vit_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from vit_keras import vit

# ===================== CONFIG =====================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# If you already have preprocessed .npy arrays (as seen in your npy_files folder),
# this script will use them directly. Otherwise it falls back to loading images
# from IMAGE_DIR (expects subfolders per class).
NPY_DIR = "npy_files"
IMAGE_DIR = "gaussian_filtered_images"

VIT_MODEL_PATH = "model/dr_vit_model.h5"

# =====================================================


def load_from_npy():
    x_train = np.load(os.path.join(NPY_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(NPY_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(NPY_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(NPY_DIR, "y_test.npy"))
    return x_train, y_train, x_test, y_test


def load_from_folders():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.15
    )
    train_gen = datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )
    return train_gen, val_gen


def build_vit_model():
    # ViT-B16 pretrained on imagenet21k, without its original classification head
    base_model = vit.vit_b16(
        image_size=IMG_SIZE,
        activation="softmax",
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    base_model.trainable = False  # freeze pretrained transformer layers first

    x = base_model.output
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    model = build_vit_model()
    model.summary()

    os.makedirs(os.path.dirname(VIT_MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(VIT_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    if os.path.isdir(NPY_DIR) and os.path.exists(os.path.join(NPY_DIR, "X_train.npy")):
        print("Loading preprocessed data from npy_files/ ...")
        x_train, y_train, x_test, y_test = load_from_npy()

        if y_train.ndim == 1:
            y_train = to_categorical(y_train, NUM_CLASSES)
            y_test = to_categorical(y_test, NUM_CLASSES)

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
        )
    else:
        print("npy files not found, loading images from folders ...")
        train_gen, val_gen = load_from_folders()
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
        )

    print(f"\nModel saved at: {VIT_MODEL_PATH}")


if __name__ == "__main__":
    main()
