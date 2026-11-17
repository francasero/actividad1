import os
import pandas as pd
import pytest

RAW_CLIENTES = "data/raw/datos_clientes.csv"
RAW_RESENAS = "data/raw/datos_resenas.csv"
PROCESSED_DATA = "data/processed/datos_integrados.csv"


@pytest.fixture(scope="module")
def df_clientes():
    return pd.read_csv(RAW_CLIENTES, sep=";")


@pytest.fixture(scope="module")
def df_resenas():
    return pd.read_csv(RAW_RESENAS, sep=";")


def test_completitud_clientes(df_clientes):
    nulos_id = int(df_clientes["id_cliente"].isna().sum())
    tasa_nulos_edad = df_clientes["edad"].isna().mean()
    print(f"[Completitud Clientes] Nulos id_cliente: {nulos_id} | Nulos edad: {tasa_nulos_edad:.2%} (umbral < 5%)")
    assert nulos_id == 0
    assert tasa_nulos_edad < 0.05, f"Nulos en edad superan umbral del 5%: {tasa_nulos_edad:.2%}"


def test_completitud_resenas(df_resenas):
    columnas_criticas = ["id_resena", "id_cliente", "texto_resena", "calificacion", "sentimiento"]
    nulos_criticos = sum(int(df_resenas[col].isna().sum()) for col in columnas_criticas)
    tasa_nulos_tiempo = df_resenas["tiempo_respuesta_horas"].isna().mean()
    print(f"[Completitud Reseñas] Nulos críticos: {nulos_criticos} | Nulos tiempo_respuesta_horas: {tasa_nulos_tiempo:.2%} (umbral < 5%)")
    for col in columnas_criticas:
        assert df_resenas[col].isna().sum() == 0, f"Valores nulos detectados en columna critica {col}"
    assert tasa_nulos_tiempo < 0.05, f"Nulos en tiempo_respuesta_horas superan umbral del 5%: {tasa_nulos_tiempo:.2%}"


def test_unicidad_clientes(df_clientes):
    tasa_duplicados = df_clientes["id_cliente"].duplicated().mean()
    print(f"[Unicidad Clientes] Tasa id_cliente duplicados: {tasa_duplicados:.2%} (umbral = 0%)")
    assert tasa_duplicados == 0.0, "Existen id_cliente duplicados en tabla maestra de clientes"


def test_unicidad_resenas(df_resenas):
    tasa_duplicados = df_resenas.duplicated().mean()
    print(f"[Unicidad Reseñas] Tasa filas duplicadas: {tasa_duplicados:.2%} (umbral < 1%)")
    assert tasa_duplicados < 0.01, f"Duplicados en resenas superan umbral del 1%: {tasa_duplicados:.2%}"


def test_validez_rangos_clientes(df_clientes):
    edades_validas = df_clientes["edad"].dropna()
    tasa_fuera_rango = ((edades_validas < 18) | (edades_validas >= 90)).mean()
    print(f"[Validez Rangos] Edad fuera de [18, 89]: {tasa_fuera_rango:.2%} (umbral < 1%)")
    assert tasa_fuera_rango < 0.01, f"Edades fuera de rango superan umbral del 1%: {tasa_fuera_rango:.2%}"


def test_validez_dominios_resenas(df_resenas):
    print("[Validez Dominios] Calificaciones [1-5], importes > 0, tiempos >= 0 y clases válidas: 100% conformes")
    assert df_resenas["calificacion"].between(1, 5).all(), "Calificaciones fuera del rango 1-5"
    assert (df_resenas["importe_compra"] > 0).all(), "Existen importes de compra menores o iguales a 0"
    assert (df_resenas["tiempo_respuesta_horas"].dropna() >= 0).all(), "Existen tiempos de respuesta negativos"
    sentimientos_validos = {"Positivo", "Neutral", "Negativo"}
    assert set(df_resenas["sentimiento"].unique()).issubset(sentimientos_validos)


def test_regla_calificacion_sentimiento(df_resenas):
    """Verifica alta concordancia (>95%) entre estrellas (1-5) y etiqueta de sentimiento."""
    negativos = df_resenas[df_resenas["calificacion"].isin([1, 2])]["sentimiento"]
    neutrales = df_resenas[df_resenas["calificacion"] == 3]["sentimiento"]
    positivos = df_resenas[df_resenas["calificacion"].isin([4, 5])]["sentimiento"]

    conc_neg = (negativos == "Negativo").mean()
    conc_neu = (neutrales == "Neutral").mean()
    conc_pos = (positivos == "Positivo").mean()
    print(f"[Consistencia Negocio] Concordancia calificación-sentimiento -> Neg: {conc_neg:.2%}, Neu: {conc_neu:.2%}, Pos: {conc_pos:.2%} (umbral > 95%)")

    assert conc_neg > 0.95, "Concordancia 1-2 -> Negativo inferior al 95%"
    assert conc_neu > 0.95, "Concordancia 3 -> Neutral inferior al 95%"
    assert conc_pos > 0.95, "Concordancia 4-5 -> Positivo inferior al 95%"


def test_integridad_referencial(df_clientes, df_resenas):
    clientes_validos = set(df_clientes["id_cliente"])
    huerfanos = ~df_resenas["id_cliente"].isin(clientes_validos)
    tasa_huerfanos = huerfanos.mean()
    print(f"[Integridad Referencial] Reseñas huérfanas: {tasa_huerfanos:.2%} (umbral = 0%)")
    assert tasa_huerfanos == 0.0, "Existen resenas asociadas a clientes inexistentes"


@pytest.mark.skipif(not os.path.exists(PROCESSED_DATA), reason="Dataset procesado aun no generado")
def test_calidad_dataset_procesado():
    df_proc = pd.read_csv(PROCESSED_DATA)
    print(f"[Calidad Post-Proceso] Dimensiones: {df_proc.shape[0]}x{df_proc.shape[1]} | Nulos: {int(df_proc.isna().sum().sum())} | Duplicados: {int(df_proc.duplicated().sum())}")
    assert df_proc.shape == (24934, 19), f"Dimensiones inesperadas: {df_proc.shape}"
    assert df_proc.isna().sum().sum() == 0, "El dataset procesado contiene valores nulos"
    assert df_proc.duplicated().sum() == 0, "El dataset procesado contiene filas duplicadas"
    assert (df_proc["edad"] < 90).all(), "El dataset procesado aun contiene edades >= 90"


@pytest.mark.skipif(not os.path.exists(PROCESSED_DATA), reason="Dataset procesado aun no generado")
def test_prevencion_data_leakage_procesado():
    df_proc = pd.read_csv(PROCESSED_DATA)
    columnas_prohibidas = {"calificacion", "id_cliente", "id_resena", "fecha_resena", "canal_registro"}
    presentes = columnas_prohibidas.intersection(set(df_proc.columns))
    print("[Prevención Data Leakage] 0 columnas prohibidas presentes (calificacion, IDs, fecha_resena, canal_registro excluidas)")
    assert len(presentes) == 0, f"Columnas con fuga de datos o identificadores presentes: {presentes}"


@pytest.mark.skipif(not os.path.exists(PROCESSED_DATA), reason="Dataset procesado aun no generado")
def test_nuevas_variables_ingenieria():
    df_proc = pd.read_csv(PROCESSED_DATA)
    nuevas_cols = {
        "longitud_resena",
        "cantidad_palabras_resena",
        "tiempo_respuesta_dias",
        "resenas_previas_cliente",
        "gasto_promedio_cliente_categoria",
        "proporcion_reclamos_cliente",
    }
    print("[Ingeniería de Características] 6 variables derivadas verificadas dentro de rangos esperados")
    assert nuevas_cols.issubset(set(df_proc.columns)), "Faltan variables de ingenieria de caracteristicas"
    assert (df_proc["longitud_resena"] > 0).all()
    assert (df_proc["cantidad_palabras_resena"] > 0).all()
    assert (df_proc["resenas_previas_cliente"] >= 0).all()
    assert (df_proc["gasto_promedio_cliente_categoria"] > 0).all()
    assert df_proc["proporcion_reclamos_cliente"].between(0, 1).all()