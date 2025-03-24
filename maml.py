import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size=3, padding=1, use_maxpool=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True)
    ]
    if use_maxpool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_filters=64):
        """
        A simple 4-layer convolutional network.
        """
        super(ConvNet, self).__init__()
        self.conv1 = conv_block(in_channels, num_filters)
        self.conv2 = conv_block(num_filters, num_filters)
        self.conv3 = conv_block(num_filters, num_filters)
        self.conv4 = conv_block(num_filters, num_filters)
        # Use adaptive pooling to ensure fixed feature dimension regardless of input size.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x, params=None):
        """
        If params is None, use the module's own parameters.
        Otherwise, use the provided parameter dictionary (for fast adaptation).
        """
        if params is None:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
        else:
            # Functional forward pass using the parameters from the dict.
            # Note: This assumes the same architecture as defined above.
            x = F.conv2d(x, params['conv1.0.weight'], params['conv1.0.bias'], stride=1, padding=1)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = F.conv2d(x, params['conv2.0.weight'], params['conv2.0.bias'], stride=1, padding=1)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = F.conv2d(x, params['conv3.0.weight'], params['conv3.0.bias'], stride=1, padding=1)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = F.conv2d(x, params['conv4.0.weight'], params['conv4.0.bias'], stride=1, padding=1)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if params is None:
            logits = self.fc(x)
        else:
            logits = F.linear(x, params['fc.weight'], params['fc.bias'])
        return logits

class MAML:
    def __init__(self, model, update_lr, num_updates):
        """
        Args:
            model: the ConvNet to be adapted.
            update_lr: inner-loop learning rate.
            num_updates: number of inner gradient update steps.
        """
        self.model = model
        self.update_lr = update_lr
        self.num_updates = num_updates
        self.loss_fn = nn.CrossEntropyLoss()

    def meta_loss(self, support_x, support_y, query_x, query_y):
        meta_batch_size = support_x.size(0)
        total_query_loss = 0
        total_accuracy = 0

        # Process each task in the meta-batch separately.
        for i in range(meta_batch_size):
            s_x = support_x[i]  # shape: [ways*shots, C, H, W]
            s_y = support_y[i]  # shape: [ways*shots]
            q_x = query_x[i]  # shape: [ways*queries, C, H, W]
            q_y = query_y[i]  # shape: [ways*queries]

            # Initialize parameters for this task.
            params = {name: param for name, param in self.model.named_parameters()}

            # Compute loss on the support set.
            logits = self.model.forward(s_x, params)
            loss = self.loss_fn(logits, s_y)

            # Inner-loop adaptation.
            for _ in range(self.num_updates):
                grads = torch.autograd.grad(loss, params.values(), create_graph=True)
                params = {name: param - self.update_lr * grad
                          for (name, param), grad in zip(params.items(), grads)}
                logits = self.model.forward(s_x, params)
                loss = self.loss_fn(logits, s_y)

            # Compute query loss and accuracy using the adapted parameters.
            query_logits = self.model.forward(q_x, params)
            q_loss = self.loss_fn(query_logits, q_y)
            preds = torch.argmax(query_logits, dim=1)
            task_accuracy = (preds == q_y).float().mean()

            total_query_loss += q_loss
            total_accuracy += task_accuracy

        avg_query_loss = total_query_loss / meta_batch_size
        avg_accuracy = total_accuracy / meta_batch_size
        return avg_query_loss, avg_accuracy
