import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# 1. CARGAR DATASET
# ---------------------------------------------------

path = r"C:\Users\josep\.cache\kagglehub\datasets\lakshmi25npathi\imdb-dataset-of-50k-movie-reviews\versions\1"
csv_path = os.path.join(path, "IMDB Dataset.csv")

df = pd.read_csv(csv_path)

print(df.head())
print("\nInformación general:")
print(df.info())
print("\nDistribución original:")
print(df["sentiment"].value_counts())


# ---------------------------------------------------
# 2. USAR MUESTRA PARA PRUEBAS RÁPIDAS
# ---------------------------------------------------

df = df.sample(10000, random_state=42)

print("\nTamaño después de sample:")
print(df.shape)


# ---------------------------------------------------
# 3. CONVERTIR SENTIMIENTO A BINARIO
# positive = 1, negative = 0
# ---------------------------------------------------

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("\nDistribución después de conversión:")
print(df["sentiment"].value_counts())


# ---------------------------------------------------
# 4. LIMPIEZA DE TEXTO
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["review_clean"] = df["review"].apply(clean_text)


# ---------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------

X = df["review_clean"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=5,
    stratify=y
)

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)


# ---------------------------------------------------
# 6. STOPWORDS
# ---------------------------------------------------

nltk.download("stopwords")
stop_words = stopwords.words("english")


# ---------------------------------------------------
# 7. FUNCIÓN DE EVALUACIÓN
# ---------------------------------------------------

def evaluate_model(model_name, model, X_train_vec, X_test_vec, y_train, y_test):
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


# ---------------------------------------------------
# 8. MODELOS RÁPIDOS
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        random_state=5,
        max_depth=30
    )
}

results = []


# ---------------------------------------------------
# 9. COUNT VECTORIZER
# ---------------------------------------------------

count_vectorizer = CountVectorizer(
    stop_words=stop_words,
    max_features=10000,
    ngram_range=(1, 1)
)

X_train_cv = count_vectorizer.fit_transform(X_train)
X_test_cv = count_vectorizer.transform(X_test)

print("\nCountVectorizer shape:", X_train_cv.shape)

for model_name, model in models.items():
    result = evaluate_model(
        f"CountVectorizer + {model_name}",
        model,
        X_train_cv,
        X_test_cv,
        y_train,
        y_test
    )
    results.append(result)


# ---------------------------------------------------
# 10. TF-IDF
# ---------------------------------------------------

tfidf_vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    max_features=10000,
    ngram_range=(1, 1)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("\nTF-IDF shape:", X_train_tfidf.shape)

for model_name, model in models.items():
    result = evaluate_model(
        f"TF-IDF + {model_name}",
        model,
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test
    )
    results.append(result)


# ---------------------------------------------------
# 11. TABLA FINAL DE RESULTADOS
# ---------------------------------------------------

results_df = pd.DataFrame(results)

print("\nRESULTADOS FINALES:")
print(results_df.sort_values(by="Accuracy", ascending=False))


# ---------------------------------------------------
# 12. GRÁFICA DE ACCURACY
# ---------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.xticks(rotation=45, ha="right")
plt.title("Comparación de modelos para Sentiment Analysis")
plt.ylabel("Accuracy")
plt.xlabel("Modelo")
plt.tight_layout()
plt.show()


# ---------------------------------------------------
# 13. MATRIZ DE CONFUSIÓN DEL MEJOR MODELO
# TF-IDF + Logistic Regression
# ---------------------------------------------------

best_model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=42
)

best_model.fit(X_train_tfidf, y_train)

y_pred_best = best_model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix - TF-IDF + Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
