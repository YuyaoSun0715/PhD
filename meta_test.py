import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_generator import MetaDataset
from maml import ConvNet, MAML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/padded_test', help='Path to meta-test dataset folder')
    parser.add_argument('--ways', type=int, default=2, help='Number of classes per task')
    parser.add_argument('--shots', type=int, default=1, help='Number of support examples per class')
    parser.add_argument('--queries', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--num_updates', type=int, default=1, help='Number of inner-loop updates for adaptation')
    parser.add_argument('--update_lr', type=float, default=0.4, help='Inner loop learning rate')
    parser.add_argument('--num_tasks', type=int, default=1000, help='Number of meta-test tasks to evaluate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    args = parser.parse_args()

    # Define the same transforms as used in meta-training.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.240], std=[0.221])
    ])

    # Load meta-test dataset from the test folder.
    test_dataset = MetaDataset(root_dir=args.data_root, ways=args.ways, shots=args.shots, queries=args.queries,
                               transform=transform)
    # Using a batch size of 1 because each item from MetaDataset is a complete task.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize model architecture for meta-testing (number of classes = args.ways)
    model = ConvNet(in_channels=1, num_classes=args.ways, num_filters=64).to(device)

    # Load the pretrained MAML model checkpoint.
    checkpoint_path = 'checkpoints/maml_model.pth'
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     print("Loaded pretrained MAML model.")
    # else:
    #     print("No pretrained model found at", checkpoint_path)
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

    maml = MAML(model, update_lr=args.update_lr, num_updates=args.num_updates)

    total_accuracy = 0.0
    count = 0

    # Evaluate the model over a fixed number of meta-test tasks.
    model.eval()
    # not wrap this part in a no-grad context.
    # Then, after adapting, we switch to no_grad mode for evaluating the query set,
    # which is common practice to save memory and speed up inference.
    # with torch.no_grad(): this has been move to line 96 to restructure the process
    for i, (support_x, support_y, query_x, query_y) in enumerate(test_loader):
        if i >= args.num_tasks:
            break

        # Remove extra batch dimension.
        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # Initialize parameters for the task (make sure they require grad).
        params = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}

        # Compute loss on support set.
        logits = model.forward(support_x, params)
        loss = maml.loss_fn(logits, support_y)

        # Inner-loop adaptation (gradient computation enabled).
        for _ in range(args.num_updates):
            grads = torch.autograd.grad(loss, params.values(), create_graph=False)
            params = {name: param - args.update_lr * grad
                      for (name, param), grad in zip(params.items(), grads)}
            logits = model.forward(support_x, params)
            loss = maml.loss_fn(logits, support_y)

        # Evaluate on the query set without gradient tracking.
        with torch.no_grad():
            query_logits = model.forward(query_x, params)
            preds = torch.argmax(query_logits, dim=1)
            accuracy = (preds == query_y).float().mean().item()
            total_accuracy += accuracy
            count += 1

    avg_accuracy = total_accuracy / count if count > 0 else 0.0
    print(f"Meta-test evaluated on {count} tasks. Average Query Accuracy: {avg_accuracy:.4f}")


if __name__ == '__main__':
    main()