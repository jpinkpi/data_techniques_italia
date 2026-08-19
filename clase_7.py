# PASAR DE MLP A CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy


# =========================
# 1. Configuración
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

transform = transforms.ToTensor()

# =========================
# 2. Cargar dataset MNIST
# =========================
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =========================
# 3. Visualizar imágenes
# =========================
plt.figure(figsize=(10, 10))
for i in range(49):
    image, label = train_dataset[i]
    plt.subplot(7, 7, i + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"{label}")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# =========================
# 4. Definir modelo CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capas fully connected
        self.fc1 = nn.Linear(32 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Entrada: (batch, 1, 28, 28)

        x = self.pool(F.relu(self.conv1(x)))   # (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (batch, 32, 7, 7)

        x = torch.flatten(x, start_dim=1)      # (batch, 32*7*7)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # logits, sin softmax

        return x

model = SimpleCNN().to(device)
print(model)

# =========================
# 5. Función de pérdida y optimizador
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 6. Entrenamiento
# =========================
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {train_accuracy:.2f}%")

# =========================
# 7. Evaluación en test
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Accuracy en test: {test_accuracy:.2f}%")

'''
## Inspect the output vector
The network produces a **vector of 10 numbers**.
Each number corresponds to the probability that the image belongs to that digit class.
'''


image, label = test_dataset[0]

with torch.no_grad():
    probs = model(image.unsqueeze(0))

probs = probs.numpy()[0]

print("Probabilities:", probs)
print("Predicted digit:", np.argmax(probs))
print("True digit:", label)
