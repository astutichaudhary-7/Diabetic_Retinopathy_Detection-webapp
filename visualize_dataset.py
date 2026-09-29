import os
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\admin\Desktop\INTERNSHIP\Diabetic Retinopathy Detection Project\gaussian_filtered_images"

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]

counts = []
for cls in classes:
    folder = os.path.join(DATASET_PATH, cls)
    counts.append(len(os.listdir(folder)))
plt.figure(figsize=(8,5))
plt.bar(classes, counts)
plt.title("Diabetic Retinopathy Dataset Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=20)
plt.show()