import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def generate_synthetic_data(num_samples=32, input_dim=10):
    """Generate synthetic regression dataset."""
    X = torch.randn(num_samples, input_dim)
    true_weights = torch.randn(input_dim, 1)
    y = X @ true_weights + torch.randn(num_samples, 1) * 0.1
    return X, y


def train_with_full_batch(model, X, y, epochs=1, batch_size=32):
    """Train model with standard full batch updates."""
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Store initial gradients for comparison
    initial_grads = [
        param.grad.clone() if param.grad is not None else None
        for param in model.parameters()
    ]

    for epoch in range(epochs):
        # Reset gradients
        optimizer.zero_grad()

        # Full batch forward and backward
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()

        # Update parameters
        optimizer.step()

    return initial_grads, model


def train_with_gradient_accumulation(model,
                                     X,
                                     y,
                                     epochs=1,
                                     batch_size=8,
                                     accumulation_steps=4):
    """Train model with gradient accumulation for memory-constrained scenarios."""
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Store initial gradients for comparison
    initial_grads = [
        param.grad.clone() if param.grad is not None else None
        for param in model.parameters()
    ]

    for epoch in range(epochs):
        # Reset gradients
        optimizer.zero_grad()

        # Simulate microbatching with gradient accumulation
        for i in range(0, len(X), batch_size):
            # Get microbatch
            micro_X = X[i:i + batch_size]
            micro_y = y[i:i + batch_size]

            # Forward and backward on microbatch
            outputs = model(micro_X)
            loss = criterion(outputs, micro_y) / accumulation_steps
            loss.backward()

        optimizer.step()

    return initial_grads, model


def check_params(m1, m2):
    for (name1, param1), (name2, param2) in zip(m1.named_parameters(),
                                                m2.named_parameters()):
        print(f'checking: {name1=} {name2=}')
        print(param1)
        print(param2)
        diff = torch.abs(param1 - param2)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        # Define tolerance for gradient comparison
        tolerance = 1e-6

        if max_diff > tolerance:
            print(f'{name1=} {name2=} {max_diff=} {mean_diff=}')
            raise RuntimeError(f'...')


def compare_gradients(m1, m2):
    for (name1, param1), (name2, param2) in zip(m1.named_parameters(),
                                                m2.named_parameters()):
        print(f'checking: {name1=} {name2=}')
        print(param1.grad)
        print(param2.grad)
        if param1.grad is None or param2.grad is None:
            continue

        # Compare gradient values
        grad_diff = torch.abs(param1.grad - param2.grad)
        max_diff = torch.max(grad_diff)
        mean_diff = torch.mean(grad_diff)
        # Define tolerance for gradient comparison
        tolerance = 1e-6

        if max_diff > tolerance:
            print(f'{name1=} {name2=} {max_diff=} {mean_diff=}')


def main():
    # Generate synthetic data
    X, y = generate_synthetic_data(32, 10)

    # Create model
    model1 = SimpleModel()
    model2 = SimpleModel()
    model2.load_state_dict(model1.state_dict())
    check_params(model1, model2)

    print(f'*' * 100 + 'params check pass')

    # Train with full batch
    print("Training with Full Batch:")
    full_batch_initial_grads, m1 = train_with_full_batch(model1, X, y)

    # Train with gradient accumulation
    print("\nTraining with Gradient Accumulation:")
    grad_accum_initial_grads, m2 = train_with_gradient_accumulation(
        model2, X, y)

    # Compare final gradients
    print("\nGradient Comparison:")
    compare_gradients(m1, m2)


if __name__ == '__main__':
    main()
