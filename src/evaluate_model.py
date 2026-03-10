import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import HindiCNN
from dataset import get_datasets


def main():

    # -------------------------
    # DEVICE
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # LOAD DATASET
    # -------------------------
    train_dataset, test_dataset = get_datasets("data/raw")

    classes = train_dataset.classes
    num_classes = len(classes)

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # -------------------------
    # LOAD MODEL
    # -------------------------
    model = HindiCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("models/hindi_cnn_best.pth", weights_only=True))
    model.to(device)
    model.eval()

    # -------------------------
    # STORE PREDICTIONS
    # -------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # -------------------------
    # METRICS
    # -------------------------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\nModel Evaluation")
    print("-----------------------")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    print("\nDetailed Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")

    print("\nConfusion matrix saved as: confusion_matrix.png")

   # SAVING RESULTS
    with open("results/evaluation_metrics.txt", "w") as f: 
       f.write("Model Evaluation\n")
       f.write("----------------------\n")
       f.write(f"Accuracy  : {acc*100:.2f}%\n")
       f.write(f"Precision : {precision:.4f}\n")
       f.write(f"Recall    : {recall:.4f}\n")
       f.write(f"F1 Score  : {f1:.4f}\n")


if __name__ == "__main__":
    main()