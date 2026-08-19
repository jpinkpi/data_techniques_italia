# ANALISIS DE SENTIMIENTO
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definir un conjunto de datos pequeño
train_data = [
    ("I love this movie", 1),
    ("This film is great", 1),
    ("Amazing acting and story", 1),
    # ("I do not hate this movie", 1), # -> Limitación de BoW
    # ("This film is not great", 0),
    ("I hate this movie", 0),
    ("This film is terrible", 0),
    ("Awful plot and acting", 0),
]
print(f"train_data es un objeto: {type(train_data)}")

# 2. Preprocesamiento: crear vocabulario
def tokenize(sentence):
    return sentence.lower().split()

# Construir vocabulario
vocab = {}
for sentence, _ in train_data:
    for word in tokenize(sentence):
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)

print(vocab)

def vectorize(sentence):
    vec = [0] * vocab_size
    for word in tokenize(sentence):
        if word in vocab:
            vec[vocab[word]] = 1  # Bolsa de palabras: presencia
    print(vec)
    return vec

sentence = 'I hate this movie'
res = vectorize(sentence)
sentence = 'I hate these movies'
res = vectorize(sentence)

X_train = torch.tensor([vectorize(sentence) for sentence, _ in train_data], dtype=torch.float32)
y_train = torch.tensor([label for _, label in train_data], dtype=torch.float32).unsqueeze(1)

torch.tensor([label for _, label in train_data], dtype=torch.float32)

# 4. Definir un modelo lineal simple
model = nn.Sequential(
    nn.Linear(vocab_size, 1),
    nn.Sigmoid()
)

# 5. Función de pérdida y optimizador
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 6. Bucle de entrenamiento
for epoch in range(50):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Época {epoch+1}, Pérdida: {loss.item():.4f}")

# 7. Función de predicción
def predict_sentiment(text_x):
    with torch.no_grad():
        prob = model(test_x)
    return "Positiva" if prob.item() >= 0.5 else "Negativa"

# 8. Ejemplos de prueba
test_sentence = "This story is great"
test_x = torch.tensor(vectorize(test_sentence), dtype=torch.float32)
print(f"La frase '{test_sentence}' es {predict_sentiment(test_x)}")

test_sentence = "The story was awful"
test_x = torch.tensor(vectorize(test_sentence), dtype=torch.float32)
print(f"La frase '{test_sentence}' es {predict_sentiment(test_x)}")


'''
Sesión práctica
Reemplaza el pequeño conjunto de entrenamiento de nuestro ejemplo con el 
conjunto de reseñas de películas de IMDB usando el dataset de Hugging Face.

Intenta:
Sustituir el dataset de prueba por 200 reseñas de IMDB (100 positivas / 100 negativas)
Entrenar el modelo y reportar la precisión en un conjunto de validación (held-out set)
Encontrar tres frases que evidencien las limitaciones del modelo y explicar 
por qué fallan utilizando el análisis BoW (Bag of Words)
'''

from datasets import load_dataset

# 1. Cargar el dataset IMDB desde Hugging Face
dataset = load_dataset("imdb")

# 2. Filtrar y seleccionar 100 ejemplos positivos y 100 negativos
positive_reviews = [item for item in dataset["train"] if item["label"] == 1][:100]  # Seleccionar 100 reseñas positivas
negative_reviews = [item for item in dataset["train"] if item["label"] == 0][:100]  # Seleccionar 100 reseñas negativas

# Ejemplo
print("Texto de las reseñas positivas: \n")
print(positive_reviews[0]['text'])
print("Etiquetas de las reseñas positivas: \n")
print(positive_reviews[0]['label'])



"""
Sustituye el conjunto de datos de prueba por 200 reseñas de IMDB (100 positivas / 100 negativas)

Entrena el modelo y reporta la precisión en un conjunto de validación (held-out set)

Encuentra tres frases que evidencien las limitaciones del modelo y explica por qué fallan utilizando el análisis BoW (Bolsa de Palabras)

"""
# Combinar reseñas positivas y negativas
train_data = positive_reviews + negative_reviews

