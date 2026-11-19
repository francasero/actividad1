import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preparacion import create_attributes

load_dotenv(find_dotenv())  # en local lee el .env; en la nube las variables ya vienen definidas
logger = logging.getLogger("uvicorn.error")  # estos mensajes salen en los logs

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")
MODEL_URI = os.getenv("MODEL_URI", "models:/sentimiento-resenas/1")


class Resena(BaseModel):
    """Campos que necesita el modelo (los atributos derivados se calculan dentro)."""

    texto_resena: str = Field(min_length=1)
    categoria_producto: Literal[
        "Alimentos", "Belleza", "Calzado", "Deportes", "Electrónica", "Hogar", "Indumentaria", "Libros"
    ]
    canal_compra: Literal["Online", "Offline"]
    plataforma: Literal[
        "Amazon", "Falabella", "Flipkart", "JioMart", "Meesho", "MercadoLibre", "Nykaa", "Zepto"
    ]
    importe_compra: float = Field(gt=0)
    tiempo_respuesta_horas: float = Field(gt=0)
    problema_resuelto: Literal["Sí", "No"]
    reclamo_formal: Literal["Sí", "No"]
    edad: float = Field(ge=18, le=90)
    genero: Literal["Femenino", "Masculino", "Otro"]
    ciudad: Literal[
        "Buenos Aires", "Córdoba", "La Plata", "Mar del Plata", "Mendoza",
        "Posadas", "Rosario", "Salta", "San Miguel de Tucumán", "Santa Fe",
    ]
    antiguedad_cliente_meses: int = Field(ge=0)


class Prediccion(BaseModel):
    sentimiento: str
    probabilidad: dict[str, float]


modelo = {}


def cargar_modelo():
    """Busca el modelo en el archivo local y, si no está, en el registro de MLflow."""
    if os.path.exists(MODEL_PATH):
        modelo["pipeline"] = joblib.load(MODEL_PATH)
        modelo["origen"] = MODEL_PATH
    elif os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow.sklearn

        modelo["pipeline"] = mlflow.sklearn.load_model(MODEL_URI)
        modelo["origen"] = MODEL_URI


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()  # se carga una sola vez, al arrancar
    yield
    modelo.clear()


app = FastAPI(
    title="API de Clasificación de Sentimientos",
    description="Predice el sentimiento (Positivo, Neutral o Negativo) de reseñas de productos.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def raiz():
    return {"message": "API de Clasificación de Sentimientos. Usa POST /predict (documentación en /docs)."}


@app.get("/health")
def health():
    if "pipeline" not in modelo:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")
    return {"status": "ok", "origen_modelo": modelo["origen"]}


@app.post("/predict", response_model=Prediccion)
def predict(resena: Resena):
    if "pipeline" not in modelo:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")

    inicio = time.perf_counter()
    pipeline = modelo["pipeline"]
    df = create_attributes(pd.DataFrame([resena.model_dump()]))  # mismos atributos que al entrenar

    sentimiento = pipeline.predict(df)[0]
    probabilidades = dict(zip(pipeline.classes_, pipeline.predict_proba(df)[0].round(4).tolist()))

    # Registro de la predicción (sin el texto de la reseña, por privacidad)
    logger.info(
        "prediccion=%s prob_max=%.3f latencia_ms=%.1f categoria=%s plataforma=%s",
        sentimiento, max(probabilidades.values()),
        (time.perf_counter() - inicio) * 1000, resena.categoria_producto, resena.plataforma,
    )
    return Prediccion(sentimiento=sentimiento, probabilidad=probabilidades)