# main.py
import argparse
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_generator import MetaDataset
from maml import ConvNet, MAML

# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='Path to dataset folder')
    parser.add_argument('--ways', type=int, default=3, help='Number of classes per task')
    parser.add_argument('--shots', type=int, default=1, help='Number of support examples per class')
    parser.add_argument('--queries', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--meta_batch_size', type=int, default=25, help='Number of tasks per meta-update')
    parser.add_argument('--num_updates', type=int, default=1, help='Number of inner-loop updates')
    parser.add_argument('--update_lr', type=float, default=0.4, help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Meta learning rate')
    parser.add_argument('--num_iterations', type=int, default=2000, help='Number of meta-training iterations') # originally 60000
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args = parser.parse_args()

    # Create dataset and dataloader.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # this transformation is NECESSARY
                                                        # Because in PyTorch, when you load images using libraries like ImageFolder and ToTensor(),
                                                        # the image is typically converted into a tensor
                                                        # where the number of channels corresponds to the image's color format.
                                                        # the grayscale images are internally represented as RGB images. This is common when working with image formats like PNG or JPEG.
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.240, std=0.221) # use your dataset's mean/std if available
    ])
    dataset = MetaDataset(root_dir=args.data_root, ways=args.ways, shots=args.shots, queries=args.queries, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.meta_batch_size, shuffle=True, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Assume input images are RGB and resized (e.g., 84x84). It's a common practice to tradeoff between feature  Adjust as needed.
    in_channels = 1
    model = ConvNet(in_channels, args.ways, num_filters=64).to(device)
    print("Model is on device:", next(model.parameters()).device)
    maml = MAML(model, update_lr=args.update_lr, num_updates=args.num_updates)

    meta_optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)

    print("Starting meta-training...")
    for iteration in range(args.num_iterations):
        # Each batch consists of meta_batch_size tasks.
        support_x, support_y, query_x, query_y = next(iter(dataloader))
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        meta_optimizer.zero_grad()
        query_loss, query_accuracy = maml.meta_loss(support_x, support_y, query_x, query_y)
        query_loss.backward()
        meta_optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Query Loss: {query_loss.item():.4f}, Query Acc: {query_accuracy.item():.4f}")

    # Save model checkpoint.
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/maml_model.pth')
    print("Meta-training complete. Model saved to checkpoints/maml_model.pth")


if __name__ == '__main__':
    main()
