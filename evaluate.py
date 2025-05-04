from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time
import os


def evaluate_model(model, val_ds):
    y_true = []
    y_pred = []

    start_time = time.time()

    for images, labels in val_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    inference_time = time.time() - start_time

    acc = accuracy_score(y_true, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print("✅ Confusion matrix saved to results/")
    print("Inference Time (s):", round(inference_time, 2))
