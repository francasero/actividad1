import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import time

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import os
import mlflow.sklearn

import mlflow
from dotenv import find_dotenv, load_dotenv

import json
from pathlib import Path

import joblib

DATA_PATH = "data/processed/datos_integrados.csv"
TARGET = "sentimiento"
COL_TEXTO = "texto_resena"
SEMILLA = 42
EXPERIMENTO = "AP2 - Sentimiento en reseñas - Modelado"
TFIDF_MAX_FEATURES = 300

DESCRIPCIONES = {
    "baseline_dummy": "Línea base: predice siempre la clase mayoritaria.",
    "logreg": "Regresión logística sin balanceo (modelo de la experimentación previa).",
    "logreg_balanceada": "Regresión logística con class_weight='balanced' para compensar el desbalanceo.",
    "logreg_balanceada_C0.1": "Regresión logística balanceada con más regularización (C=0.1).",
    "logreg_balanceada_C10": "Regresión logística balanceada con menos regularización (C=10).",
    "linear_svc": "SVM lineal. No entrega probabilidades.",
    "knn": "K vecinos más cercanos (k=5).",
    "arbol_decision": "Árbol de decisión sin poda.",
    "random_forest_balanceado": "Bosque aleatorio de 150 árboles con class_weight='balanced'.",
}

COLUMNAS_NO_DISPONIBLES = [
    "resenas_previas_cliente",
    "gasto_promedio_cliente_categoria",
    "proporcion_reclamos_cliente",
]


def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[TARGET] + COLUMNAS_NO_DISPONIBLES), df[TARGET]


def dividir_datos(X, y):
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEMILLA
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.22, stratify=y_rest, random_state=SEMILLA
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def crear_preprocesador(X):
    categoricas = [c for c in X.select_dtypes(include="object").columns if c != COL_TEXTO]
    numericas = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("texto", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES), COL_TEXTO),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numericas),
    ])

def crear_candidatos():
    return {
        "baseline_dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000),
        "logreg_balanceada": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "logreg_balanceada_C0.1": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1),
        "logreg_balanceada_C10": LogisticRegression(max_iter=2000, class_weight="balanced", C=10),
        "linear_svc": LinearSVC(max_iter=5000),
        "knn": KNeighborsClassifier(),
        "arbol_decision": DecisionTreeClassifier(random_state=SEMILLA),
        "random_forest_balanceado": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight="balanced",
            random_state=SEMILLA, n_jobs=-1,
        ),
    }


def construir_pipeline(estimador, X):
    return Pipeline([("prep", crear_preprocesador(X)), ("model", clone(estimador))])


def calcular_metricas(y_true, y_pred, proba=None, clases=None):
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_negativo": recall_score(y_true, y_pred, labels=["Negativo"], average=None)[0],
    }
    if proba is not None:
        metricas["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=clases)
    return metricas

def configurar_mlflow():
    load_dotenv(find_dotenv())
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENTO)

if __name__ == "__main__":
    configurar_mlflow()
    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)
    pipelines = {}
    filas = []
    for nombre, estimador in crear_candidatos().items():
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tag("descripcion", DESCRIPCIONES[nombre])
            mlflow.log_params({"modelo": type(estimador).__name__, **estimador.get_params()})
            mlflow.log_params({
                "tfidf_max_features": TFIDF_MAX_FEATURES,
                "semilla": SEMILLA,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
            })

            pipeline = construir_pipeline(estimador, X_train)
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            segundos = time.perf_counter() - t0

            pred = pipeline.predict(X_val)
            proba = pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
            m = calcular_metricas(y_val, pred, proba, pipeline.classes_)

            mlflow.log_metrics({f"val_{k}": float(v) for k, v in m.items()})
            mlflow.log_metric("segundos_entrenamiento", segundos)
        filas.append({"candidato": nombre, "segundos": segundos, **m})
        pipelines[nombre] = pipeline

    tabla = pd.DataFrame(filas).sort_values("f1_macro", ascending=False)
    print(tabla.round(3).to_string(index=False))
    # El mejor por F1-macro entre los que dan probabilidades (la tabla ya está ordenada)
    elegibles = tabla[tabla["roc_auc_ovr"].notna()]
    nombre_mejor = elegibles.iloc[0]["candidato"]
    print("\nMejor modelo:", nombre_mejor)

    # Evaluación en test (una sola vez)
    pipeline = pipelines[nombre_mejor]
    pred_test = pipeline.predict(X_test)
    proba_test = pipeline.predict_proba(X_test)
    m_test = calcular_metricas(y_test, pred_test, proba_test, pipeline.classes_)
    print({k: round(float(v), 3) for k, v in m_test.items()})
    # Run final: registrar el modelo en el Model Registry de MLflow
    with mlflow.start_run(run_name=f"MODELO FINAL - {nombre_mejor}"):
        mlflow.set_tag("etapa", "modelo_final")
        mlflow.log_params({"modelo": type(pipeline.named_steps["model"]).__name__,
                           **pipeline.named_steps["model"].get_params()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m_test.items()})
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="sentimiento-resenas",
            serialization_format="cloudpickle",
        )
    # Guardar el modelo y sus métricas
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/trained_model.pkl")
    resumen = {"modelo_seleccionado": nombre_mejor}
    resumen.update({f"test_{k}": float(v) for k, v in m_test.items()})
    with open("metrics/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)