import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
RETINA_DIR = "gaussian_filtered_images"
RANDOM_DIR = "random_images"
CIFAR_NEGATIVE_COUNT = 4000
GATEKEEPER_MODEL_PATH = "model/gatekeeper_model.h5"
def load_retina_paths(retina_dir):
    paths = []
    for root, _, files in os.walk(retina_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    return paths
def load_random_paths(random_dir):
    paths = []
    if os.path.isdir(random_dir):
        for f in os.listdir(random_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(random_dir, f))
    return paths
def load_cifar_negatives(n_samples):
    print("Loading CIFAR-100 (auto-downloads on first run, ~170MB)...")
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data()
    all_images = np.concatenate([x_train, x_test], axis=0)
    n_samples = min(n_samples, len(all_images))
    idx = np.random.choice(len(all_images), size=n_samples, replace=False)
    selected = all_images[idx]
    resized = tf.image.resize(selected, (IMG_SIZE, IMG_SIZE)).numpy()
    return resized  # float32, values 0-255
def img_path_to_array(path):
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return img_to_array(img)
def build_dataset():
    retina_paths = load_retina_paths(RETINA_DIR)
    random_paths = load_random_paths(RANDOM_DIR)
    print(f"Retina images found:        {len(retina_paths)}")
    print(f"Your own random images:     {len(random_paths)}")
    if len(retina_paths) == 0:
        raise RuntimeError(
            f"No retina images found in '{RETINA_DIR}'. Check RETINA_DIR path."
        )
    X, y = [], []
    for p in retina_paths:
        X.append(img_path_to_array(p))
        y.append(1.0)  # 1 = retina
    for p in random_paths:
        X.append(img_path_to_array(p))
        y.append(0.0)  # 0 = not retina
    cifar_negatives = load_cifar_negatives(CIFAR_NEGATIVE_COUNT)
    for img in cifar_negatives:
        X.append(img)
        y.append(0.0)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")

    print(f"Total samples: {len(X)}  |  Retina: {int(y.sum())}  |  Not-retina: {int(len(y) - y.sum())}")
    return X, y
def build_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze pretrained layers first
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary: retina (1) vs not (0)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
def main():
    X, y = build_dataset()
    X = preprocess_input(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    model = build_model()
    model.summary()

    os.makedirs(os.path.dirname(GATEKEEPER_MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(GATEKEEPER_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {val_acc:.4f}")
    print(f"Gatekeeper model saved at: {GATEKEEPER_MODEL_PATH}")


if __name__ == "__main__":
    main()
