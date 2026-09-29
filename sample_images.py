import os
import random

import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = r"C:\Users\admin\Desktop\INTERNSHIP\Diabetic Retinopathy Detection Project\gaussian_filtered_images"

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]

plt.figure(figsize=(15,5))

for i, cls in enumerate(classes):

    folder = os.path.join(DATASET_PATH, cls)

    image_file = random.choice(os.listdir(folder))

    image_path = os.path.join(folder, image_file)

    img = Image.open(image_path)

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    plt.title(cls)

    plt.axis("off")

plt.show()