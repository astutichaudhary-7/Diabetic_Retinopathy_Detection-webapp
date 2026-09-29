import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
model = load_model("dr_model.h5")
predictions = model.predict(X_test)
y_pred = predictions.argmax(axis=1)

class_names = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]

print(classification_report(y_test, y_pred, target_names=class_names))