import torch
from torch.utils.data import DataLoader

from model import HindiCNN
from dataset import get_datasets


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load dataset
    train_dataset, test_dataset = get_datasets("data/raw")

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # load model
    model = HindiCNN(num_classes=len(train_dataset.classes))
    model.load_state_dict(torch.load("models/hindi_cnn_best.pth", weights_only=True))
    model.to(device)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("\nTest Accuracy:", f"{accuracy:.2f}%")


if __name__ == "__main__":
    main()