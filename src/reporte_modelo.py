from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # dibuja sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.train_model import cargar_datos, dividir_datos

REPORT_DIR = Path("reports/modelo")

if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    modelo = joblib.load("models/trained_model.pkl")
    X, y = cargar_datos()
    X_test, y_test = dividir_datos(X, y)[2], dividir_datos(X, y)[5]
    pred = modelo.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=modelo.classes_, cmap="Blues")
    plt.title("Matriz de confusión (conjunto de test)")
    plt.savefig(REPORT_DIR / "confusion_matrix_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    informe = classification_report(y_test, pred, labels=modelo.classes_, digits=3)
    (REPORT_DIR / "informe_clasificacion_test.txt").write_text(informe, encoding="utf-8")
    print(informe)