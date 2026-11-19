import os

import pandas as pd
import requests
import streamlit as st

AUTORES = "Francisco Javier Casero Burgos"


def obtener_api_url():
    """Prioridad: variable de entorno > secreto de Streamlit Cloud > API local."""
    url = os.getenv("API_URL")
    if not url:
        try:
            url = st.secrets["API_URL"]
        except Exception:
            url = "http://127.0.0.1:8000"
    return url.rstrip("/")


API_URL = obtener_api_url()

st.set_page_config(page_title="Sentimiento de reseñas", page_icon="💬")
st.title("Predicción de sentimiento de reseñas")
st.caption(f"Autor: {AUTORES}")

with st.form("formulario"):
    texto = st.text_area("Texto de la reseña")
    col1, col2 = st.columns(2)
    categoria = col1.selectbox(
        "Categoría del producto",
        ["Alimentos", "Belleza", "Calzado", "Deportes", "Electrónica", "Hogar", "Indumentaria", "Libros"],
    )
    canal = col2.selectbox("Canal de compra", ["Online", "Offline"])
    plataforma = col1.selectbox(
        "Plataforma",
        ["Amazon", "Falabella", "Flipkart", "JioMart", "Meesho", "MercadoLibre", "Nykaa", "Zepto"],
    )
    importe = col2.number_input("Importe de la compra", min_value=0.01, value=20000.0)
    tiempo = col1.number_input("Tiempo de respuesta (horas)", min_value=0.01, value=24.0)
    resuelto = col2.selectbox("¿Se resolvió el problema?", ["Sí", "No"])
    reclamo = col1.selectbox("¿Hubo reclamo formal?", ["No", "Sí"])
    edad = col2.number_input("Edad del cliente", min_value=18, max_value=90, value=30)
    genero = col1.selectbox("Género", ["Femenino", "Masculino", "Otro"])
    ciudad = col2.selectbox(
        "Ciudad",
        ["Buenos Aires", "Córdoba", "La Plata", "Mar del Plata", "Mendoza",
         "Posadas", "Rosario", "Salta", "San Miguel de Tucumán", "Santa Fe"],
    )
    antiguedad = col1.number_input("Antigüedad del cliente (meses)", min_value=0, value=12)
    enviar = st.form_submit_button("Predecir")

if enviar:
    if not texto.strip():
        st.warning("Escribe el texto de la reseña.")
    else:
        datos = {
            "texto_resena": texto,
            "categoria_producto": categoria,
            "canal_compra": canal,
            "plataforma": plataforma,
            "importe_compra": importe,
            "tiempo_respuesta_horas": tiempo,
            "problema_resuelto": resuelto,
            "reclamo_formal": reclamo,
            "edad": edad,
            "genero": genero,
            "ciudad": ciudad,
            "antiguedad_cliente_meses": int(antiguedad),
        }
        try:
            with st.spinner("Consultando a la API (puede tardar un minuto si estaba dormida)..."):
                respuesta = requests.post(f"{API_URL}/predict", json=datos, timeout=90)
            if respuesta.status_code == 200:
                resultado = respuesta.json()
                st.subheader(f"Sentimiento estimado: {resultado['sentimiento']}")
                st.bar_chart(pd.Series(resultado["probabilidad"]))
                if resultado["sentimiento"] == "Negativo":
                    st.error("Reseña negativa: conviene priorizar su atención.")
            else:
                st.error(f"La API devolvió un error ({respuesta.status_code}): {respuesta.text}")
        except requests.exceptions.RequestException:
            st.error(f"No se pudo conectar con la API ({API_URL}).")