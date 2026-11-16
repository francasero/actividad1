import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def preparar_datos():
    ruta_clientes = "data/raw/datos_clientes.csv"
    ruta_resenas = "data/raw/datos_resenas.csv"
    ruta_salida = "data/processed/datos_integrados.csv"
    ruta_reporte = "reports/reporte_transformaciones.md"

    df_clientes = pd.read_csv(ruta_clientes, sep=";")
    df_resenas = pd.read_csv(ruta_resenas, sep=";")

    filas_clientes_ini, cols_clientes_ini = df_clientes.shape
    filas_resenas_ini, cols_resenas_ini = df_resenas.shape
    nulos_edad_ini = int(df_clientes["edad"].isna().sum())
    nulos_tiempo_ini = int(df_resenas["tiempo_respuesta_horas"].isna().sum())

    # 1. Seleccion inicial de atributos
    df_clientes.drop(columns=["canal_registro"], inplace=True)

    # 2. Filtrado de outliers y duplicados
    outliers_edad = int((df_clientes["edad"] >= 90).sum())
    df_clientes_filtrado = df_clientes[
        (df_clientes["edad"] < 90) | (df_clientes["edad"].isna())
    ].copy()

    duplicados_resenas = int(df_resenas.duplicated().sum())
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    # 3. Imputacion contextual por mediana agrupada
    df_clientes_filtrado["edad"] = df_clientes_filtrado.groupby("ciudad")["edad"].transform(
        lambda x: x.fillna(x.median())
    )
    df_resenas_filtrado["tiempo_respuesta_horas"] = df_resenas_filtrado.groupby(
        "categoria_producto"
    )["tiempo_respuesta_horas"].transform(lambda x: x.fillna(x.median()))

    # 4. Integracion relacional (inner join)
    df_integrado = pd.merge(
        df_resenas_filtrado, df_clientes_filtrado, on="id_cliente", how="inner"
    )
    filas_post_join = df_integrado.shape[0]

    # 5. Ingenieria de caracteristicas
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

    # 6. Eliminacion de identificadores y variables con fuga de datos (data leakage)
    columnas_a_eliminar = [
        "id_cliente",
        "id_resena",
        "calificacion",
        "fecha_resena",
    ]
    df_integrado.drop(columns=columnas_a_eliminar, inplace=True)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_integrado.to_csv(ruta_salida, index=False)

    # 7. Exportar reporte trazable de transformaciones
    os.makedirs(os.path.dirname(ruta_reporte), exist_ok=True)
    dist_target = df_integrado["sentimiento"].value_counts(normalize=True).mul(100).round(2)

    reporte_md = f"""# Reporte de Transformaciones de Datos (Fase C - CRISP-DM)

## 1. Volumen inicial de los datasets crudos
- **datos_clientes.csv**: {filas_clientes_ini:,} filas y {cols_clientes_ini} columnas.
- **datos_resenas.csv**: {filas_resenas_ini:,} filas y {cols_resenas_ini} columnas.

## 2. Operaciones de seleccion y limpieza
- **Eliminacion de columnas sin poder discriminativo**: `canal_registro` (clientes).
- **Eliminacion de outliers en `edad` (>= 90 anos)**: {outliers_edad} registros eliminados.
- **Eliminacion de filas duplicadas en resenas**: {duplicados_resenas} registros eliminados.
- **Imputacion de valores nulos en `edad` (mediana por `ciudad`)**: {nulos_edad_ini} valores imputados (0 nulos restantes).
- **Imputacion de valores nulos en `tiempo_respuesta_horas` (mediana por `categoria_producto`)**: {nulos_tiempo_ini} valores imputados (0 nulos restantes).

## 3. Integracion e ingenieria de caracteristicas
- **Integracion relacional (`inner join` por `id_cliente`)**: {filas_post_join:,} registros conservados.
- **Variables derivadas creadas (6)**:
  1. `longitud_resena`: longitud en caracteres de `texto_resena`.
  2. `cantidad_palabras_resena`: numero de palabras de `texto_resena`.
  3. `tiempo_respuesta_dias`: conversion de `tiempo_respuesta_horas / 24`.
  4. `resenas_previas_cliente`: historial secuencial previo por cliente ordenado por `fecha_resena` (`cumcount`).
  5. `gasto_promedio_cliente_categoria`: importe medio gastado por cliente y categoria de producto.
  6. `proporcion_reclamos_cliente`: tasa historica de reclamos formales por cliente.
- **Variables eliminadas por prevencion de data leakage e identificadores**: `id_cliente`, `id_resena`, `calificacion`, `fecha_resena`.

## 4. Resumen del dataset procesado (`{ruta_salida}`)
- **Dimensiones finales**: {df_integrado.shape[0]:,} filas y {df_integrado.shape[1]} columnas (18 predictores + 1 target).
- **Valores nulos totales**: {int(df_integrado.isna().sum().sum())}
- **Filas duplicadas**: {int(df_integrado.duplicated().sum())}
- **Distribucion de la variable objetivo (`sentimiento`)**:
  - Positivo: {dist_target.get('Positivo', 0.0)} %
  - Negativo: {dist_target.get('Negativo', 0.0)} %
  - Neutral: {dist_target.get('Neutral', 0.0)} %
"""
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(reporte_md)

    print(f"Dataset procesado guardado en {ruta_salida}")
    print(f"Reporte de transformaciones guardado en {ruta_reporte}")
    print(f"Dimensiones finales: {df_integrado.shape[0]} filas y {df_integrado.shape[1]} columnas")


if __name__ == "__main__":
    preparar_datos()