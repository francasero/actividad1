# Predicción de Sentimiento en Reseñas de Clientes (13MBID)

Proyecto desarrollado sobre la plantilla de la asignatura **13MBID - Metodologías de Gestión de Proyectos de Big Data** (*Máster Universitario en Big Data y Ciencia de Datos, Universidad Internacional de Valencia* — Prof. Dr. Horacio Kuna), aplicando el marco metodológico **CRISP-DM + Scrum + DataOps**.

- **Repositorio de código (GitHub):** https://github.com/francasero/actividad1
- **Repositorio de datos y artefactos (DagsHub):** https://dagshub.com/francasero/actividad1
- **Tablero Scrum (GitHub Projects):** https://github.com/users/francasero/projects/1/views/1

## Descripción del contenido

- `config/`: archivos de configuración y dependencias del entorno (`uv_requirements.txt`, entre otros) documentados en [Entornos.md](Entornos.md).
- `data/`: almacenamiento de datos versionados mediante DVC y DagsHub:
  - `raw/`: datasets crudos originales (`datos_clientes.csv.dvc` y `datos_resenas.csv.dvc`).
  - `processed/`: dataset limpio, integrado y enriquecido (`datos_integrados.csv`).
- `notebooks/`: libretas de exploración y procesamiento inicial (`CD01_Visualizacion.ipynb` y `CD02_Procesamiento.ipynb`).
- `src/`: scripts modulares ejecutados por el pipeline de DVC:
  - `visualizacion.py`: generación de gráficas univariantes y bivariadas en `reports/figures/`.
  - `perfilado.py`: generación de informes automáticos con `ydata-profiling` en `reports/perfilado/`.
  - `preparacion.py`: limpieza, imputación contextual, ingeniería de características y exportación de `reports/reporte_transformaciones.md`.
- `tests/`: batería de pruebas automatizadas de calidad e integridad de datos (`test_calidad_datos.py`).
- `reports/`: artefactos generados por el pipeline (figuras, perfilado HTML, reporte de transformaciones y `reporte_calidad.html`).
- `docs/` y `examples/`: documentación de referencia y ejemplos de verificación del entorno de la plantilla base.
- `dvc.yaml`, `dvc.lock` y `uv.lock`: definición declarativa del pipeline reproducible (4 etapas: `visualizacion`, `perfilado`, `preparacion` y `calidad`) y bloqueo de dependencias.

## Cómo reproducir el proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/francasero/actividad1.git
   cd actividad1
   ```

2. **Replicar y activar el entorno de Python** (consulta [Entornos.md](Entornos.md) para más opciones):

   Con **uv** (recomendado):
   ```bash
   uv sync
   ```

   Con **venv + pip**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install -r config/uv_requirements.txt
   ```

3. **Descargar los datos versionados (DVC + DagsHub S3):**

El repositorio utiliza un bucket S3 compatible en DagsHub configurado bajo el remoto `storage`. Para descargar los datasets crudos, los datos procesados y el modelo versionado, configura tus credenciales de DagsHub en local (sin exponerlas en Git) y ejecuta `dvc pull`:

   ```bash
   dvc remote modify --local storage access_key_id <TU_TOKEN_O_ACCESS_KEY_DAGSHUB>
   dvc remote modify --local storage secret_access_key <TU_TOKEN_O_SECRET_KEY_DAGSHUB>
   dvc pull -r storage
   ```

4. **Ejecutar el pipeline completo de forma reproducible** (incluye visualización, perfilado, preparación y puerta de calidad):
   ```bash
   dvc repro
   dvc dag
   ```

5. **Ejecutar la batería de pruebas de calidad de forma independiente (opcional):**
   ```bash
   pytest tests/test_calidad_datos.py -v --html=reports/reporte_calidad.html --self-contained-html
   ```