import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -------------------------
# 1. Cargar datos
# -------------------------
df = pd.read_csv(r"C:\Users\josep\Downloads\creditcard.csv")

normal_df = df[df["Class"] == 0]
fraud_df = df[df["Class"] == 1]

X_train = normal_df.drop("Class", axis=1).values
X_test = df.drop("Class", axis=1).values
y_test = df["Class"].values

# -------------------------
# 2. Escalado
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reducir tamaño si quieres hacer pruebas rápidas
X_train = X_train[:20000]
X_test = X_test[:10000]
y_test = y_test[:10000]

# -------------------------
# 3. Tensores
# -------------------------
Xtr = torch.tensor(X_train, dtype=torch.float32)
Xte = torch.tensor(X_test, dtype=torch.float32)

# -------------------------
# 4. Autoencoder
# -------------------------
class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, d)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# -------------------------
# 5. Modelo, optimizador, loss
# -------------------------
model = AE(Xtr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -------------------------
# 6. Entrenamiento
# -------------------------
epochs = 20

for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    
    recon = model(Xtr)
    loss = loss_fn(recon, Xtr)
    
    loss.backward()
    opt.step()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

# -------------------------
# 7. Error de reconstrucción
# -------------------------
model.eval()
with torch.no_grad():
    recon_train = model(Xtr)
    train_err = ((recon_train - Xtr) ** 2).mean(dim=1).numpy()

    recon_test = model(Xte)
    test_err = ((recon_test - Xte) ** 2).mean(dim=1).numpy()

# -------------------------
# 8. Umbral
# -------------------------
threshold = train_err.mean() + 3 * train_err.std()
print("Threshold:", threshold)

# Predicciones
y_pred = (test_err > threshold).astype(int)

# -------------------------
# 9. Evaluación
# -------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nROC-AUC:", roc_auc_score(y_test, test_err))