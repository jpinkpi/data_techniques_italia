# -*- coding: utf-8 -*-
"""Word2Vec preentrenado + IMDB

Pipeline:
Texto -> tokens -> Word2Vec (300d) -> promedio -> clasificador lineal

No entrenamos Word2Vec. 
Alguien ya leyó medio mundo por nosotros… nosotros solo juzgamos.
"""

# 1. Imports
import numpy as np
import gensim.downloader as gensim_dl
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Cargar embeddings (la “memoria colectiva” del lenguaje)
print("Descargando Word2Vec (paciencia, ~1.6GB)...")
w2v = gensim_dl.load("word2vec-google-news-300")
d = w2v.vector_size

print(f"Dimensión: {d} | Vocabulario: {len(w2v.key_to_index):,}")

# Verificación: la geometría semántica existe (o eso queremos creer)
print("\nAnalogía: king - man + woman ≈ ?")
for w, s in w2v.most_similar(positive=["king","woman"], negative=["man"], topn=3):
    print(f"{w:<12} cos={s:.4f}")

# 3. Dataset IMDB (emociones comprimidas en etiquetas binarias… qué ironía)
dataset = load_dataset("imdb")
train_raw, test_raw = dataset["train"], dataset["test"]

# 4. Tokenización (simple, casi ingenua)
def tok(t): return t.lower().split()

# 5. Embedding promedio (aquí empieza la simplificación peligrosa)
def embed(t):
    vecs = [w2v[w] for w in tok(t) if w in w2v]
    return np.mean(vecs, axis=0) if vecs else np.zeros(d)

print("Construyendo embeddings… (sí, esto tarda)")
X_train = np.array([embed(x["text"]) for x in train_raw])
y_train = np.array([x["label"] for x in train_raw])

X_test  = np.array([embed(x["text"]) for x in test_raw])
y_test  = np.array([x["label"] for x in test_raw])

# 6. Clasificador (el juez: frío, lineal, eficaz… y ciego a matices)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7. Evaluación
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# 8. Inferencia (donde el modelo revela sus limitaciones)
def pred(t):
    p = clf.predict_proba(embed(t).reshape(1,-1))[0,1]
    lab = "Positivo" if p>=0.5 else "Negativo"
    print(f"[{lab:8s} p={p:.3f}] {t}")

print("\n--- Casos normales ---")
pred("This movie was amazing")
pred("Terrible and boring film")

print("\n--- Negación (fallo clásico) ---")
pred("not good at all")
pred("not bad at all")

print("\n--- Polisemia (misma palabra, distinto mundo) ---")
pred("I went to the bank and deposited money")
pred("We sat on the bank of the river")

print("\n--- Orden ignorado (aquí el modelo se pierde) ---")
pred("The film was good not bad")
pred("The film was bad not good")

# 9. Comparación BoW (el viejo método que se resiste a morir)
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(max_features=20000, binary=True)
Xb_train = vec.fit_transform([x["text"] for x in train_raw])
Xb_test  = vec.transform([x["text"] for x in test_raw])

clf_bow = LogisticRegression(max_iter=1000)
clf_bow.fit(Xb_train, y_train)

acc_bow = accuracy_score(y_test, clf_bow.predict(Xb_test))

print("\nComparación:")
print(f"BoW: {acc_bow:.4f} | W2V: {acc:.4f}")

# 10. Interpretación de pesos (la “opinión” del modelo)
w = clf.coef_[0]
words = ["amazing","love","perfect","terrible","boring","hate","good","bad"]

print("\nPalabras y su carga emocional según el modelo:")
for wd in words:
    if wd in w2v:
        score = float(w @ w2v[wd])
        print(f"{wd:<10} {score:+.3f}")

"""
Resumen del maestro (con una sonrisa leve):

- Word2Vec da geometría al lenguaje.
- Promediar destruye contexto.
- El modelo clasifica… pero no comprende.

Como diría :contentReference[oaicite:1]{index=1}:
funciona, pero no sabe por qué.
"""