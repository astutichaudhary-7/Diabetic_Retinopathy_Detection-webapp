import os

DATASET_PATH = r"C:\Users\admin\Desktop\INTERNSHIP\Diabetic Retinopathy Detection Project\gaussian_filtered_images"

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]

print("\nImages Per Class\n")

for cls in classes:

    folder = os.path.join(DATASET_PATH, cls)

    total = len(os.listdir(folder))

    print(f"{cls} : {total}")