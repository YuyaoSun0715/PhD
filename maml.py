import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MAML(nn.Module):
    def __init__(self, dim_input=1, dim_output=1, hidden_layers=[40, 40], update_lr=0.01, meta_lr=0.001, num_updates=5):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.hidden_layers = hidden_layers
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.num_updates = num_updates

        self.network = self.build_network()
        self.loss_fn = nn.MSELoss()
        self.meta_optimizer = optim.Adam(self.network.parameters(), lr=self.meta_lr)

    def build_network(self):
        layers = []
        input_dim = self.dim_input
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.dim_output))
        return nn.Sequential(*layers)

    def forward(self, x, weights=None):
        if weights is None:
            return self.network(x)
        else:
            out = x
            for i in range(0, len(weights) - 2, 2):
                out = torch.matmul(out, weights[i]) + weights[i + 1]
                out = torch.relu(out)
            return torch.matmul(out, weights[-2]) + weights[-1]

    def clone_weights(self):
        return [param.clone() for param in self.network.parameters()]

    def task_adaptation(self, x, y):
        fast_weights = self.clone_weights()
        for _ in range(self.num_updates):
            preds = self.forward(x, fast_weights)
            loss = self.loss_fn(preds, y)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.update_lr * g for w, g in zip(fast_weights, grads)]
        return fast_weights

    def meta_update(self, task_batch):
        meta_loss = 0.0
        for x_train, y_train, x_test, y_test in task_batch:
            fast_weights = self.task_adaptation(x_train, y_train)
            preds = self.forward(x_test, fast_weights)
            meta_loss += self.loss_fn(preds, y_test)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()


# Example usage
maml = MAML(dim_input=1, dim_output=1)
x = torch.randn(5, 1)
y = torch.randn(5, 1)
task_batch = [(x, y, x, y)]  # Example dummy task batch
meta_loss = maml.meta_update(task_batch)
print("Meta Loss:", meta_loss)
