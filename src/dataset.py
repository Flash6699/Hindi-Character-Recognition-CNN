from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_datasets(data_dir):

    transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = ImageFolder(
        root=f"{data_dir}/Train",
        transform=transform
    )

    test_dataset = ImageFolder(
        root=f"{data_dir}/Test",
        transform=transform
    )

    return train_dataset, test_dataset