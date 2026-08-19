#EJERCICIO DE CLASE 11
'''
Replicar lo visto en clase 1o
a un dataset diferente(FashionMNIST dataset)'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


## Load the MNIST dataset
# 1. Config and download (MNIST)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_dim = 2
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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


# 2. Definition of VAE model
class VAE(nn.Module):

    # init defines how the encoder and the decoder work
    # here the encoder is a two layer map 784 -> 400 -> 2
    # and the dencoder is a two layer map 2 -> 400 -> 784
    # in both cases activation functions are ReLU(f) = max(f,0) for intermediate layers
    # and Sigmoid(f) = 1 / ( 1+exp(f) ) for the output layer as pixel values are in [0,1]
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()

        # encoder definition:  convulational layer
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1), # 28x28 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 14x14 -> 7x7
        
            #second layer is a linear layer that maps the output of the convolutional layer to the latent space
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 7x7 -> 3x3
        
            #Third layer produces latent space parameters (mean and log variance)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 3x3 -> 3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 3x3 -> 1x1
        )

        # encoder definition: second layer has two stream, i.e. mean and variance
        self.flattened_size = 128 * 3 * 3
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        # Output size of the convolutional layers after flattening

        # decoder definition: first and second layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    # apply the encoder
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten the tensor
        return self.fc_mu(h), self.fc_logvar(h)

    # reparametrization trick
    # sampling one point from the distribution with
    # mean : mu
    # variance : var
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # std = variance squared root (0.5)
        eps = torch.randn_like(std) # randomization
        return mu + eps*std

    # apply the decoder
    def decode(self, z):
        return self.decoder(z)

    # Apply the encoder + reparametrization  trick
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)  # output are randomized
        return recon_x, mu, logvar

model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 4. Fast demo training
print("Start training...")
model.train()
for epoch in range(15):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch+1}, average loss: {train_loss / len(train_loader.dataset):.4f}")


# 5. Generation of new digits
def plot_latent_space(model, scale=1.0, n=20):
    # Grid of 2D points in the latent space
    # n number of images
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)
    figure = np.zeros((28 * n, 28 * n))

    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                x_decoded = model.decoder(z_sample)
                digit = x_decoded.view(28, 28).cpu().numpy()
                figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title("VAE latent space : move among digits")
    plt.xlabel("Dimension Z1")
    plt.ylabel("Dimension Z2")
    plt.show()

plot_latent_space(model, scale=1.0)


def plot_distribution(model, data_loader):
    model.eval()
    z_values = []
    with torch.no_grad():
        for data, _ in data_loader:
            mu, logvar = model.encode(data.to(device))
            z = model.reparameterize(mu, logvar)
            z_values.append(z.cpu().numpy())

    z_values = np.concatenate(z_values, axis=0)

    plt.figure(figsize=(8, 4))
    plt.hist(z_values[:, 0], bins=50, alpha=0.5, label='Dimension Z1')
    plt.hist(z_values[:, 1], bins=50, alpha=0.5, label='Dimension Z2')
    plt.title("Global distribution in the latent space (Target: Gaussian)")
    plt.legend()
    plt.show()

# Eseguila per vedere la "Campana di Gauss" formata dai dati
plot_distribution(model, train_loader)