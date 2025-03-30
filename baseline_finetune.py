import argparse
import os
import copy
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from torchvision import transforms
from data_generator import MetaDataset
from maml import ConvNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/padded_test', help='Path to meta-test dataset folder')
    parser.add_argument('--ways', type=int, default=2, help='Number of classes per task')
    parser.add_argument('--shots', type=int, default=1, help='Number of support examples per class')
    parser.add_argument('--queries', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--finetune_steps', type=int, default=1, help='Number of fine-tuning steps on the support set')
    parser.add_argument('--finetune_lr', type=float, default=1e-3, help='Learning rate for fine-tuning')
    parser.add_argument('--num_tasks', type=int, default=1000, help='Number of meta-test tasks to evaluate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    args = parser.parse_args()

    # Define the same transforms used during meta-training/testing.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.240], std=[0.221])
    ])

    # Load meta-test dataset (tasks sampled from unseen classes).
    test_dataset = MetaDataset(root_dir=args.data_root, ways=args.ways, shots=args.shots, queries=args.queries, transform=transform)
    # Use batch size of 1 because each sample is an entire task.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize the conventional classifier with the same architecture (number of outputs = args.ways).
    model = ConvNet(in_channels=1, num_classes=args.ways, num_filters=64).to(device)

    # Load the pretrained conventional classifier (trained via standard supervised training).
    checkpoint_path = 'checkpoints/baseline_model.pth'
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     print("Loaded pre-trained conventional classifier.")
    # else:
    #     print("No pre-trained conventional model found at", checkpoint_path)
    #     return

    ##### However since there are only two classes for test dataset, we have to exclude the final fc layer when loading and reinitialize the final fc layer to match
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    # Exclude the fc layer from the checkpoint
    pretrained_dict = {k: v for k, v in checkpoint.items() if not k.startswith('fc')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # Reinitialize the final fully connected layer to match the current number of classes.
    import torch.nn as nn
    model.fc = nn.Linear(model.fc.in_features, args.ways).to(device)

    criterion = nn.CrossEntropyLoss()

    total_accuracy = 0.0
    count = 0

    # Evaluate over multiple meta-test tasks.
    for i, (support_x, support_y, query_x, query_y) in enumerate(test_loader):
        if i >= args.num_tasks:
            break

        # Remove the batch dimension.
        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # Make a fresh copy of the pre-trained model for fine-tuning.
        finetune_model = copy.deepcopy(model)
        finetune_model.train()

        optimizer = optim.Adam(finetune_model.parameters(), lr=args.finetune_lr)

        # Fine-tune the model on the support set.
        for _ in range(args.finetune_steps):
            optimizer.zero_grad()
            outputs = finetune_model(support_x)
            loss = criterion(outputs, support_y)
            loss.backward()
            optimizer.step()

        # Evaluate the fine-tuned model on the query set.
        finetune_model.eval()
        with torch.no_grad():
            query_outputs = finetune_model(query_x)
            preds = torch.argmax(query_outputs, dim=1)
            accuracy = (preds == query_y).float().mean().item()
            total_accuracy += accuracy
            count += 1

    avg_accuracy = total_accuracy / count if count > 0 else 0.0
    print(f"Fine-tuning baseline evaluated on {count} tasks. Average Query Accuracy: {avg_accuracy:.4f}")

if __name__ == '__main__':
    main()
