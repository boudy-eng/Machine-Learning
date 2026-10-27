import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import time
import statistics
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_full = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_set, val_set = random_split(
    train_full,
    [55_000, 5_000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

images, labels = next(iter(train_loader))
print("Train / validation / test:", len(train_set), len(val_set), len(test_set))
print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each 7×7 patch has 49 values; there are 16 patches.
        self.patch_projection = nn.Linear(49, 64)
        self.position_embedding = nn.Parameter(torch.zeros(1, 16, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]

        # [batch, 1, 28, 28] → [batch, 16 patches, 49 values]
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(batch_size, 16, 49)

        x = self.patch_projection(x) + self.position_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)  # Mean pool over the 16 patches
        return self.classifier(x)


models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

for name, model in models.items():
    model.eval()
    sample = torch.randn(1, 1, 28, 28)

    print(f"\n{name}")
    summary(model, input_size=(1, 1, 28, 28), verbose=0)

    macs, params = profile(model, inputs=(sample,), verbose=False)
    print(f"THOP MACs: {macs:,.0f}")
    print(f"THOP parameters: {params:,.0f}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTiming device: {device}")

for name, model in models.items():
    model = model.to(device)
    model.eval()

    for batch_size, repeats in [(1, 100), (128, 20)]:
        sample = images[:batch_size].to(device)

        # Warm up the model before measuring.
        with torch.inference_mode():
            for _ in range(10):
                model(sample)

        times = []
        with torch.inference_mode():
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        print(
            f"{name}, batch {batch_size}: "
            f"median {statistics.median(times):.3f} ms"
        )
        criterion = nn.CrossEntropyLoss()

for name, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining {name}")

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/5, loss: {average_loss:.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    print(f"{name} test accuracy: {100 * correct / total:.2f}%")
    import matplotlib.pyplot as plt

names = ["MLP", "CNN", "Attention"]
accuracies = [87.61, 89.67, 86.20]
parameters = [235146, 20490, 71818]

# Use the latency results from your latest run.
latency_batch_1 = [0.737, 2.263, 6.519]
latency_batch_128 = [3.223, 24.690, 29.910]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

plots = [
    (parameters, "Parameters", "Accuracy vs Parameters"),
    (latency_batch_1, "Latency (ms), batch 1", "Accuracy vs Batch 1 Latency"),
    (latency_batch_128, "Latency (ms), batch 128", "Accuracy vs Batch 128 Latency"),
]

for ax, (x_values, x_label, title) in zip(axes, plots):
    ax.scatter(x_values, accuracies)

    for i, name in enumerate(names):
        ax.annotate(name, (x_values[i], accuracies[i]))

    ax.set_xlabel(x_label)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.savefig("tradeoff_plots.png", dpi=300)
plt.show()