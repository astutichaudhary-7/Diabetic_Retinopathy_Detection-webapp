import os
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split

DATASET_PATH = r"C:\Users\admin\Desktop\INTERNSHIP\Diabetic Retinopathy Detection Project\gaussian_filtered_images"

classes = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4
}

IMG_SIZE = 224

images = []
labels = []

print("Loading Images...")

loaded = 0

for folder_name, label in classes.items():

    folder_path = os.path.join(DATASET_PATH, folder_name)

    print(f"Reading {folder_name} images...")

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        try:
            img = load_img(
                image_path,
                target_size=(IMG_SIZE, IMG_SIZE)
            )

            img = img_to_array(img)
            img = img / 255.0

            images.append(img)
            labels.append(label)

            loaded += 1

        except Exception as e:
            print("Skipped:", image_name)
            print("Reason:", e)

print("Total Loaded Images:", loaded)

images = np.array(images)
labels = np.array(labels)

print("\nDataset Loaded Successfully")
print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nTrain-Test Split Done")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nFiles Saved Successfully")