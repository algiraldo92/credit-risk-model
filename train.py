import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

"""
========================================
MODEL TRAINING: CREDIT RISK MODEL
========================================

Este script:
- Carga los datos simulados
- Aplica preprocessing
- Entrena un modelo de clasificación
- Evalúa con ROC-AUC
- Guarda el modelo listo para producción
"""

# cargar datos
df = pd.read_csv("data.csv")

# features y target
X = df.drop("default_flag", axis=1)
y = df["default_flag"]

# columnas
num_cols = [
    "customer_age",
    "monthly_income",
    "total_debt",
    "active_loans_count",
    "delinquency_count"
]

cat_cols = ["employment_type"]

# preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# entrenar
pipeline.fit(X_train, y_train)

# evaluar
probs = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print(f"ROC-AUC: {auc:.2f}")

# guardar modelo
joblib.dump(pipeline, "modelo.pkl")