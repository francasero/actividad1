# Reporte de Transformaciones de Datos (Fase C - CRISP-DM)

## 1. Volumen inicial de los datasets crudos
- **datos_clientes.csv**: 10,400 filas y 6 columnas.
- **datos_resenas.csv**: 25,063 filas y 13 columnas.

## 2. Operaciones de seleccion y limpieza
- **Eliminacion de columnas sin poder discriminativo**: `canal_registro` (clientes).
- **Eliminacion de outliers en `edad` (>= 90 anos)**: 28 registros eliminados.
- **Eliminacion de filas duplicadas en resenas**: 63 registros eliminados.
- **Imputacion de valores nulos en `edad` (mediana por `ciudad`)**: 243 valores imputados (0 nulos restantes).
- **Imputacion de valores nulos en `tiempo_respuesta_horas` (mediana por `categoria_producto`)**: 757 valores imputados (0 nulos restantes).

## 3. Integracion e ingenieria de caracteristicas
- **Integracion relacional (`inner join` por `id_cliente`)**: 24,934 registros conservados.
- **Variables derivadas creadas (6)**:
  1. `longitud_resena`: longitud en caracteres de `texto_resena`.
  2. `cantidad_palabras_resena`: numero de palabras de `texto_resena`.
  3. `tiempo_respuesta_dias`: conversion de `tiempo_respuesta_horas / 24`.
  4. `resenas_previas_cliente`: historial secuencial previo por cliente ordenado por `fecha_resena` (`cumcount`).
  5. `gasto_promedio_cliente_categoria`: importe medio gastado por cliente y categoria de producto.
  6. `proporcion_reclamos_cliente`: tasa historica de reclamos formales por cliente.
- **Variables eliminadas por prevencion de data leakage e identificadores**: `id_cliente`, `id_resena`, `calificacion`, `fecha_resena`.

## 4. Resumen del dataset procesado (`data/processed/datos_integrados.csv`)
- **Dimensiones finales**: 24,934 filas y 19 columnas (18 predictores + 1 target).
- **Valores nulos totales**: 0
- **Filas duplicadas**: 0
- **Distribucion de la variable objetivo (`sentimiento`)**:
  - Positivo: 64.52 %
  - Negativo: 17.88 %
  - Neutral: 17.6 %
