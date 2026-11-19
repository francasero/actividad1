"""Monitoreo básico del modelo (Actividad 4).

Compara un lote de reseñas nuevas (con el sentimiento real) con los datos de entrenamiento
y con los objetivos de negocio. Guarda un JSON con las métricas y las alertas.

Uso:
    python -m src.monitor data/raw/resenas_nuevas.csv metrics/monitoring_normal.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.preparacion import create_attributes

warnings.filterwarnings("ignore", category=UserWarning)

# Umbrales de alerta
UMBRAL_PSI = 0.20  # PSI > 0,2: cambio relevante en la distribución de una variable
UMBRAL_DIF_CLASES = 0.10  # más de 10 puntos de cambio en la proporción predicha de una clase
UMBRAL_DIF_RECLAMOS = 0.02  # más de 2 puntos de cambio en la tasa de reclamos formales

# Objetivos mínimos de desempeño (los mismos que en las pruebas del modelo)
OBJETIVOS = {"accuracy": 0.80, "f1_macro": 0.75, "recall_negativo": 0.80}

VARIABLES_NUMERICAS = ["tiempo_respuesta_horas", "importe_compra"]


def calcular_psi(referencia, actual, bins=10):
    """Population Stability Index entre la distribución de entrenamiento y la actual."""
    cortes = np.unique(np.quantile(referencia.dropna(), np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    pct_ref = np.histogram(referencia.dropna(), cortes)[0] / referencia.notna().sum()
    pct_act = np.histogram(actual.dropna(), cortes)[0] / actual.notna().sum()
    pct_ref = np.clip(pct_ref, 1e-6, None)
    pct_act = np.clip(pct_act, 1e-6, None)
    return float(np.sum((pct_act - pct_ref) * np.log(pct_act / pct_ref)))


def monitor(lote_path, output_path, referencia_path="data/processed/datos_integrados.csv",
            model_path="models/trained_model.pkl"):
    modelo = joblib.load(model_path)
    referencia = pd.read_csv(referencia_path)
    lote = pd.read_csv(lote_path, sep=";")

    # Mismos atributos derivados que en el entrenamiento y en la API
    X_lote = create_attributes(lote.drop(columns=["sentimiento"]))
    pred = pd.Series(modelo.predict(X_lote), index=lote.index)

    metricas = {"cantidad_resenas": int(len(lote))}
    alertas = []

    # 1) Deriva de las variables numéricas de entrada
    for col in VARIABLES_NUMERICAS:
        psi = calcular_psi(referencia[col], lote[col])
        metricas[f"psi_{col}"] = round(psi, 4)
        if psi > UMBRAL_PSI:
            alertas.append(f"Deriva en '{col}': PSI = {psi:.3f} (umbral {UMBRAL_PSI})")

    # 2) Proporción de clases predichas frente a la del entrenamiento
    dist_ref = referencia["sentimiento"].value_counts(normalize=True)
    dist_pred = pred.value_counts(normalize=True)
    for clase in dist_ref.index:
        actual = float(dist_pred.get(clase, 0.0))
        metricas[f"proporcion_predicha_{clase.lower()}"] = round(actual, 4)
        if abs(actual - dist_ref[clase]) > UMBRAL_DIF_CLASES:
            alertas.append(f"Cambio en la proporción predicha de '{clase}': {dist_ref[clase]:.1%} -> {actual:.1%}")

    # 3) Desempeño frente a los objetivos (el lote trae el sentimiento real)
    real = lote["sentimiento"]
    desempeno = {
        "accuracy": accuracy_score(real, pred),
        "f1_macro": f1_score(real, pred, average="macro"),
        "recall_negativo": recall_score(real, pred, labels=["Negativo"], average="macro"),
    }
    for nombre, umbral in OBJETIVOS.items():
        valor = round(float(desempeno[nombre]), 4)
        metricas[nombre] = valor
        if valor < umbral:
            alertas.append(f"{nombre} por debajo del objetivo: {valor:.3f} < {umbral}")

    # 4) Tasa de reclamos formales
    tasa_ref = float((referencia["reclamo_formal"] == "Sí").mean())
    tasa_act = float((lote["reclamo_formal"] == "Sí").mean())
    metricas["tasa_reclamos_formales"] = round(tasa_act, 4)
    if abs(tasa_act - tasa_ref) > UMBRAL_DIF_RECLAMOS:
        alertas.append(f"Cambio en la tasa de reclamos formales: {tasa_ref:.1%} -> {tasa_act:.1%}")

    reporte = {"lote": str(lote_path), "estado": "ALERTA" if alertas else "OK",
               "metricas": metricas, "alertas": alertas}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"Lote analizado: {lote_path} ({len(lote)} reseñas) -> estado: {reporte['estado']}")
    for clave, valor in metricas.items():
        print(f"  {clave}: {valor}")
    for alerta in alertas:
        print(f"  ALERTA: {alerta}")
    return reporte


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoreo del modelo con un lote de reseñas nuevas")
    parser.add_argument("lote", nargs="?", default="data/raw/resenas_nuevas.csv")
    parser.add_argument("salida", nargs="?", default="metrics/monitoring.json")
    parser.add_argument("--fallar-si-alerta", action="store_true",
                        help="termina con código 1 si hay alertas, para que GitHub Actions falle y avise")
    args = parser.parse_args()
    resultado = monitor(args.lote, args.salida)
    if args.fallar_si_alerta and resultado["alertas"]:
        sys.exit(1)