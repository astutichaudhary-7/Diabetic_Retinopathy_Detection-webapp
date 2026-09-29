import numpy as np
from tensorflow.keras.models import load_model

X_test=np.load("X_test.npy")
y_test=np.load("y_test.npy")

model=load_model("dr_model.h5")

loss,accuracy=model.evaluate(X_test,y_test)

print("Loss :",loss)
print("Accuracy :",accuracy)