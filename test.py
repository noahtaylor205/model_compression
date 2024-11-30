import torch
import torch.nn as nn
import torch.optim as optim
import time

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dummy dataset
input_size = 100
output_size = 10
num_samples = 10000

X = torch.randn(num_samples, input_size).to(device)
y = torch.randint(0, output_size, (num_samples,)).to(device)

# Simple model
model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, output_size)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
batch_size = 64
num_epochs = 2  # Keep epochs low for quick runs

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        # Get batch
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds on {device}")
