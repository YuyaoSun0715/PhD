import argparse
import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from maml import ConvNet  # using the same model architecture from your maml.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/train', help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--test_split', type=float, default=0.2, help='Proportion of dataset to use for testing')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    args = parser.parse_args()

    # Define transforms (same as in your main.py, adjusted for conventional training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # necessary as in your meta-learning setup
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.240], std=[0.221])
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    num_samples = len(dataset)
    test_size = int(args.test_split * num_samples)
    train_size = num_samples - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set the number of output classes to the number of classes in your dataset.
    num_classes = len(dataset.classes)
    model = ConvNet(in_channels=1, num_classes=num_classes, num_filters=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("Starting conventional training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Evaluate on the test set.
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss /= total
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Optionally, save the trained model.
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/baseline_model.pth')
    print("Conventional training complete. Model saved to checkpoints/baseline_model.pth")

if __name__ == '__main__':
    main()