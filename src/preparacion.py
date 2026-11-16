import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def preparar_datos():
    ruta_clientes = "data/raw/datos_clientes.csv"
    ruta_resenas = "data/raw/datos_resenas.csv"
    ruta_salida = "data/processed/datos_integrados.csv"

    df_clientes = pd.read_csv(ruta_clientes, sep=";")
    df_resenas = pd.read_csv(ruta_resenas, sep=";")

    df_clientes.drop(columns=["canal_registro"], inplace=True)

    df_clientes_filtrado = df_clientes[
        (df_clientes["edad"] < 90) | (df_clientes["edad"].isna())
    ].copy()
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    df_clientes_filtrado["edad"] = df_clientes_filtrado.groupby("ciudad")["edad"].transform(
        lambda x: x.fillna(x.median())
    )
    df_resenas_filtrado["tiempo_respuesta_horas"] = df_resenas_filtrado.groupby(
        "categoria_producto"
    )["tiempo_respuesta_horas"].transform(lambda x: x.fillna(x.median()))

    df_integrado = pd.merge(
        df_resenas_filtrado, df_clientes_filtrado, on="id_cliente", how="inner"
    )

    df_integrado["longitud_resena"] = df_integrado["texto_resena"].str.len()
    df_integrado["cantidad_palabras_resena"] = (
        df_integrado["texto_resena"].str.split().apply(len)
    )
    df_integrado["tiempo_respuesta_dias"] = df_integrado["tiempo_respuesta_horas"] / 24

    df_integrado["fecha_resena"] = pd.to_datetime(df_integrado["fecha_resena"])
    df_integrado = df_integrado.sort_values(
        by=["id_cliente", "fecha_resena", "id_resena"]
    ).reset_index(drop=True)

    df_integrado["resenas_previas_cliente"] = df_integrado.groupby("id_cliente").cumcount()
    df_integrado["gasto_promedio_cliente_categoria"] = df_integrado.groupby(
        ["id_cliente", "categoria_producto"]
    )["importe_compra"].transform("mean")

    df_integrado["es_reclamo"] = (
        df_integrado["reclamo_formal"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["sí", "si"])
        .astype(int)
    )
    df_integrado["proporcion_reclamos_cliente"] = df_integrado.groupby("id_cliente")[
        "es_reclamo"
    ].transform("mean")
    df_integrado.drop(columns=["es_reclamo"], inplace=True)

    columnas_a_eliminar = [
        "id_cliente",
        "id_resena",
        "calificacion",
        "fecha_resena",
    ]
    df_integrado.drop(columns=columnas_a_eliminar, inplace=True)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_integrado.to_csv(ruta_salida, index=False)

    print(f"Dataset procesado guardado en {ruta_salida}")
    print(f"Dimensiones finales: {df_integrado.shape[0]} filas y {df_integrado.shape[1]} columnas")
    print(f"Valores nulos totales: {df_integrado.isna().sum().sum()}")
    print(f"Filas duplicadas: {df_integrado.duplicated().sum()}")


if __name__ == "__main__":
    preparar_datos()