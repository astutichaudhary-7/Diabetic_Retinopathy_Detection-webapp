import numpy as np
import matplotlib.pyplot as plt

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("===== DATASET INFORMATION =====")
print("X_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)

print("\nFirst 10 Training Labels:")
print(y_train[:10])

print("\nFirst 10 Testing Labels:")
print(y_test[:10])

plt.figure(figsize=(5,5))
plt.imshow(X_train[0])
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()

print("\nFirst Image Pixel Values:")
print(X_train[0])