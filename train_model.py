import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Cargar dataset
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Precisión:", accuracy_score(y_test, preds))

joblib.dump(model, "model_shieldnet.joblib")
print("✅ Modelo guardado como model_shieldnet.joblib")
