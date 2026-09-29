import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Dataset Loaded")

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3), 
        activation='relu',
        input_shape=(224,224,3)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dense(
        5,
        activation='softmax'
    )
)

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model Compiled Successfully")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

with open("history.pkl","wb") as f:
    pickle.dump(history.history, f)

model.save("dr_model.h5")

print("Model Saved Successfully")
