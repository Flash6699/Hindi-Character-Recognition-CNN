import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HindiCNN
from dataset import get_datasets


def main():

    # -----------------------
    # DEVICE
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True

    # -----------------------
    # LOAD DATASET
    # -----------------------
    train_dataset, test_dataset = get_datasets("data/raw")

    print("Train images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Classes:", train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------
    # MODEL
    # -----------------------
    model = HindiCNN(num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    best_loss = float("inf")

    os.makedirs("models", exist_ok=True)

    # -----------------------
    # TRAINING LOOP
    # -----------------------
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # -----------------------
        # SAVE BEST MODEL
        # -----------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/hindi_cnn_best.pth")
            print("Best model saved!")

    print("\nTraining finished")
    print("Best training loss:", best_loss)
    print("Model saved at: models/hindi_cnn_best.pth")


if __name__ == "__main__":
    main()