

#Un MLP sencillo para el reconocimiento de digitos
'''
En este cuaderno reproducimos una tarea simple de reconocimiento 
similar al humano: reconocer dígitos escritos a mano.
La idea es similar a la narrativa en explicaciones modernas de IA:
Una imagen es simplemente una cuadrícula de valores de píxeles.
Aplanamos (flatten) la imagen en un vector de números.
Aplicamos multiplicaciones de matrices y no linealidades.
La red produce probabilidades para cada clase de dígito (0-9).
Nuestra arquitectura será:

imagen (28x28)
   ↓ aplanar
784 números
   ↓ capa lineal (matriz 784 x 32)
32 números
   ↓ ReLU
32 números
   ↓ capa lineal (matriz 32 x 10)
10 números (logits)
   ↓ softmax
10 probabilidades
El vector final nos dice con qué intensidad la imagen pertenece a cada clase de dígito.
'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Carga el MNIST dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

#Visualizamos algunos dígitos
image, label = train_dataset[0]

plt.figure(figsize=(10, 10))
for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_dataset[i][0].squeeze(), cmap='gray')
    plt.title(f"Label: {train_dataset[i][1]}")
plt.show()

plt.figure(figsize=(10, 15))
for i in range(15):
    plt.subplot(15, 1, i + 1)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.plot(train_dataset[i][0].flatten())
    plt.title(f"Label: {train_dataset[i][1]}")
plt.show()

'''
## Define the MLP

The model implements exactly the pipeline described earlier:

* flatten the image
* multiply by a **784 x 32 matrix**
* apply **ReLU**
* multiply by a **32 x 10 matrix**
* apply **softmax**

'''

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28*28, 100) #El segundo argumento es el número de neuronas que queremos en la capa oculta
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(100, 10) #El segundo argumento es el número de neuronas que queremos en la capa de salida
        self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

model = SimpleMLP()
print(model)

## Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 30
for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch+1, "loss:", total_loss/len(train_loader))


#Inspeccionar el vector de salida
'''
La red produce un vector de 10 números.
Cada número corresponde a la
probabilidad de que la imagen 
pertenezca a esa clase de dígito.
'''
image, label = test_dataset[0]

with torch.no_grad():
    probs = model(image.unsqueeze(0))

probs = probs.numpy()[0]

print("Probabilities:", probs)
print("Predicted digit:", np.argmax(probs))
print("True digit:", label)


#Visualización del vector de probabilidades
'''
Visualizamos el vector de salida como un mapa de calor en una fila.
La celda con el color más intenso corresponde a la clase predicha.
'''

plt.imshow(probs.reshape(1,10))
plt.yticks([])
plt.xticks(range(10))
plt.title("Softmax output (probability for each digit class)")
plt.show()

#Interpretación

'''
píxeles → representación interna → probabilidades
El vector de 10 dimensiones puede interpretarse como un embedding (representación) de la imagen en el espacio de las clases de dígitos.
Si el dígito es claramente un 4, la entrada correspondiente a la clase 4 dominará el vector.
Este flujo simple ilustra el principio clave utilizado en los sistemas modernos de IA:
las entradas de alta dimensión se transforman, a través de capas de transformaciones lineales y no lineales, en representaciones que facilitan la clasificación
'''

y_pred = []
y_test = []

model.eval()  # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient calculations during inference
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_test.extend(labels.cpu().numpy())

# Convert to numpy arrays for sklearn metrics
y_pred = np.array(y_pred)
y_test = np.array(y_test)

print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)


acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()


