import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Model Definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

# 3. Training and Evaluation Functions
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    return accuracy

# 4. Experimentation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depths = [2, 3, 4]
widths = [32, 64, 128, 256, 512]
results_task1 = {}  # {(depth, width): median_accuracy}

for depth in depths:
    for width in widths:
        accuracies = []
        for repeat in range(3):  # Repeat 3 times
            hidden_sizes = [width] * depth
            model = MLP(28*28, hidden_sizes, 10).to(device)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            for epoch in range(10):  # Train for 10 epochs
                train(model, device, train_loader, optimizer, criterion)
            accuracy = test(model, device, test_loader)
            accuracies.append(accuracy)
            print(f'Depth: {depth}, Width: {width}, Repeat: {repeat+1}, Accuracy: {accuracy:.2f}%')
        median_accuracy = np.median(accuracies)
        results_task1[(depth, width)] = median_accuracy
        print(f'Depth: {depth}, Width: {width}, Median Accuracy: {median_accuracy:.2f}%')

# 5. Plotting Results
# For each depth, plot median accuracy vs. width
plt.figure(figsize=(10, 6))
for depth in depths:
    accuracies = [results_task1[(depth, width)] for width in widths]
    plt.plot(widths, accuracies, marker='o', label=f'Depth {depth}')
plt.title('Task 1: Median Test Accuracy vs. Width for Different Depths')
plt.xlabel('Width (Number of Neurons per Layer)')
plt.ylabel('Median Test Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# 1. Compressed Linear Layer
class CompressedLinear(nn.Module):
    def __init__(self, input_size, output_size, rank):
        super(CompressedLinear, self).__init__()
        self.A = nn.Linear(input_size, rank, bias=False)
        self.B = nn.Linear(rank, output_size, bias=True)
        
    def forward(self, x):
        return self.B(self.A(x))

# 2. Compressed MLP Model
class CompressedMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, ranks):
        super(CompressedMLP, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size, rank in zip(hidden_sizes, ranks):
            layers.append(CompressedLinear(last_size, hidden_size, rank))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

# 3. Experimentation with Compression
compression_rates = [0.25, 0.5, 0.75]  # Example compression rates
results_task2 = {}  # {(depth, width, compression_rate): median_accuracy}

for depth in depths:
    for width in widths:
        for rate in compression_rates:
            accuracies = []
            for repeat in range(3):  # Repeat 3 times
                hidden_sizes = [width] * depth
                ranks = [max(1, int(width * rate))] * depth  # Ensure rank is at least 1
                model = CompressedMLP(28*28, hidden_sizes, 10, ranks).to(device)
                optimizer = optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()
                for epoch in range(10):  # Train for 10 epochs
                    train(model, device, train_loader, optimizer, criterion)
                accuracy = test(model, device, test_loader)
                accuracies.append(accuracy)
                print(f'Depth: {depth}, Width: {width}, Compression Rate: {rate}, Repeat: {repeat+1}, Accuracy: {accuracy:.2f}%')
            median_accuracy = np.median(accuracies)
            results_task2[(depth, width, rate)] = median_accuracy
            print(f'Depth: {depth}, Width: {width}, Compression Rate: {rate}, Median Accuracy: {median_accuracy:.2f}%')

# 4. Plotting Results
for depth in depths:
    plt.figure(figsize=(10, 6))
    # Plot the baseline (uncompressed) results from Task 1
    accuracies_task1 = [results_task1[(depth, width)] for width in widths]
    plt.plot(widths, accuracies_task1, marker='o', label='Uncompressed')
    
    # Plot compressed results for each compression rate
    for rate in compression_rates:
        accuracies_task2 = [results_task2[(depth, width, rate)] for width in widths]
        plt.plot(widths, accuracies_task2, marker='x', linestyle='--', label=f'Compressed (Rate={rate})')
    
    plt.title(f'Task 2: Median Test Accuracy vs. Width at Depth {depth}')
    plt.xlabel('Width (Number of Neurons per Layer)')
    plt.ylabel('Median Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 1. Define Distillation Loss Function
def distillation_loss(y_student, y_teacher, y_true, T, alpha):
    loss_fn = nn.CrossEntropyLoss()
    soft_targets = nn.functional.log_softmax(y_student / T, dim=1)
    soft_labels = nn.functional.softmax(y_teacher / T, dim=1)
    distill_loss = nn.functional.kl_div(soft_targets, soft_labels, reduction='batchmean') * (T * T)
    hard_loss = loss_fn(y_student, y_true)
    return alpha * hard_loss + (1 - alpha) * distill_loss

# 2. Training Function with Knowledge Distillation
def train_student(student_model, teacher_model, device, train_loader, optimizer, T, alpha):
    student_model.train()
    teacher_model.eval()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_output = student_model(data)
        with torch.no_grad():
            teacher_output = teacher_model(data)
        loss = distillation_loss(student_output, teacher_output, target, T, alpha)
        loss.backward()
        optimizer.step()

# 3. Train Teacher Model
best_depth = 3
best_width = 512
hidden_sizes = [best_width] * best_depth
teacher_model = MLP(28*28, hidden_sizes, 10).to(device)
optimizer = optim.Adam(teacher_model.parameters())
criterion = nn.CrossEntropyLoss()

print("Training the teacher model...")
for epoch in range(10):  # Train for 10 epochs
    train(teacher_model, device, train_loader, optimizer, criterion)
teacher_accuracy = test(teacher_model, device, test_loader)
print(f'Teacher Model Test Accuracy: {teacher_accuracy:.2f}%')

# 4. Experimentation with Knowledge Distillation
T = 2.0
alpha = 0.5
results_task3 = {}  # {(depth, width, compression_rate): median_accuracy}

for depth in depths:
    for width in widths:
        for rate in compression_rates:
            accuracies = []
            for repeat in range(3):  # Repeat 3 times
                hidden_sizes = [width] * depth
                ranks = [max(1, int(width * rate))] * depth
                student_model = CompressedMLP(28*28, hidden_sizes, 10, ranks).to(device)
                optimizer = optim.Adam(student_model.parameters())
                for epoch in range(10):  # Train for 10 epochs
                    train_student(student_model, teacher_model, device, train_loader, optimizer, T, alpha)
                accuracy = test(student_model, device, test_loader)
                accuracies.append(accuracy)
                print(f'Depth: {depth}, Width: {width}, Compression Rate: {rate}, Repeat: {repeat+1}, KD Accuracy: {accuracy:.2f}%')
            median_accuracy = np.median(accuracies)
            results_task3[(depth, width, rate)] = median_accuracy
            print(f'Depth: {depth}, Width: {width}, Compression Rate: {rate}, KD Median Accuracy: {median_accuracy:.2f}%')

# 5. Plotting Results
for depth in depths:
    for rate in compression_rates:
        plt.figure(figsize=(10, 6))
        # Plot compressed models without KD (Task 2)
        accuracies_task2 = [results_task2[(depth, width, rate)] for width in widths]
        plt.plot(widths, accuracies_task2, marker='x', linestyle='--', label='Without KD')
        
        # Plot compressed models with KD (Task 3)
        accuracies_task3 = [results_task3[(depth, width, rate)] for width in widths]
        plt.plot(widths, accuracies_task3, marker='o', linestyle='-', label='With KD')
        
        plt.title(f'Task 3: Compressed Models at Depth {depth} and Compression Rate {rate}')
        plt.xlabel('Width (Number of Neurons per Layer)')
        plt.ylabel('Median Test Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
