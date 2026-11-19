import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, recall_score

from src.train_model import calcular_metricas, cargar_datos, dividir_datos

MODEL_PATH = "models/trained_model.pkl"

# Objetivos de negocio (umbrales mínimos)
ACCURACY_MIN = 0.80
F1_MACRO_MIN = 0.75
RECALL_NEGATIVO_MIN = 0.80
RECALL_CLASE_MIN = 0.70  # ninguna clase puede quedar desatendida


@pytest.fixture(scope="module")
def modelo():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def particiones():
    X, y = cargar_datos()
    return dividir_datos(X, y)  # X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture(scope="module")
def pred_test(modelo, particiones):
    return modelo.predict(particiones[2])


@pytest.fixture(scope="module")
def metricas_test(modelo, particiones, pred_test):
    X_test, y_test = particiones[2], particiones[5]
    return calcular_metricas(y_test, pred_test, modelo.predict_proba(X_test), modelo.classes_)


def test_accuracy_cumple_objetivo(metricas_test):
    assert metricas_test["accuracy"] >= ACCURACY_MIN


def test_f1_macro_cumple_objetivo(metricas_test):
    assert metricas_test["f1_macro"] >= F1_MACRO_MIN


def test_recall_clase_negativa_cumple_objetivo(metricas_test):
    assert metricas_test["recall_negativo"] >= RECALL_NEGATIVO_MIN


def test_recall_minimo_en_todas_las_clases(modelo, particiones, pred_test):
    y_test = particiones[5]
    recalls = recall_score(y_test, pred_test, labels=modelo.classes_, average=None)
    assert all(r >= RECALL_CLASE_MIN for r in recalls)


def test_supera_a_la_linea_base(particiones, metricas_test):
    X_train, X_test, y_train, y_test = particiones[0], particiones[2], particiones[3], particiones[5]
    base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    f1_base = f1_score(y_test, base.predict(X_test), average="macro")
    assert metricas_test["f1_macro"] > f1_base + 0.20