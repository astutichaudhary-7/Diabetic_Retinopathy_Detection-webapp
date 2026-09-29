import matplotlib.pyplot as plt
import pickle

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")

plt.show()