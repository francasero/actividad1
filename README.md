# Predicción de Sentimiento en Reseñas de Clientes (13MBID)

Proyecto desarrollado sobre la plantilla de la asignatura **13MBID - Metodologías de Gestión de Proyectos de Big Data** (*Máster Universitario en Big Data y Ciencia de Datos, Universidad Internacional de Valencia* — Prof. Dr. Horacio Kuna), aplicando el marco metodológico **CRISP-DM + Scrum + DataOps**.

- **Repositorio de código (GitHub):** https://github.com/francasero/actividad1
- **Repositorio de datos y artefactos (DagsHub):** https://dagshub.com/francasero/actividad1
- **Tablero Scrum (GitHub Projects):** https://github.com/users/francasero/projects/1/views/1

## Descripción del contenido

- `config/`: archivos de configuración del entorno (`requirements.txt`, entre otros) documentados en [Entornos.md](Entornos.md).
- `data/`: almacenamiento de datos versionados mediante DVC y DagsHub:
  - `raw/`: datasets crudos originales (`datos_clientes.csv.dvc` y `datos_resenas.csv.dvc`).
  - `processed/`: dataset limpio, integrado y enriquecido (`datos_integrados.csv`).
- `notebooks/`: libretas de exploración y procesamiento inicial (`CD01_Visualizacion.ipynb` y `CD02_Procesamiento.ipynb`).
- `src/`: scripts modulares ejecutados por el pipeline de DVC:
  - `visualizacion.py`: generación de gráficas exploratorias en `reports/figures/`.
  - `perfilado.py`: generación de informes automáticos con `ydata-profiling` en `reports/perfilado/`.
  - `preparacion.py`: limpieza, imputación contextual, ingeniería de características y exportación de `reports/reporte_transformaciones.md`.
- `tests/`: batería de pruebas automatizadas de calidad e integridad de datos (`test_calidad_datos.py`).
- `reports/`: artefactos generados por el pipeline (figuras, perfilado HTML, reporte de transformaciones y `reporte_calidad.html`).
- `docs/` y `examples/`: documentación de referencia y ejemplos de verificación del entorno de la plantilla base.
- `dvc.yaml`, `dvc.lock` y `uv.lock`: definición declarativa del pipeline reproducible y bloqueo de dependencias.

## Cómo reproducir el proyecto

1. Clonar el repositorio:
  ```bash
  git clone https://github.com/francasero/actividad1.git
  cd actividad1
   ```

2. Replicar y activar el entorno de Python (consulta Entornos.md para más opciones):

  Con venv + pip (Linux/macOS):
   ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r config/requirements.txt
  ```

  Con uv:
  ```bash
  uv sync
  ```

  Con conda:
  ```bash
  conda create -n actividad1 python=3.12
  conda activate actividad1
  pip install -r config/requirements.txt
  ``` 
  
  Con pip tracicional:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # En Windows: .venv\Scripts\activate
  pip install -r config/uv_requirements.txt
  ```


3. Descargar los datos y artefactos versionados desde DagsHub (DVC):
  ```bash
  dvc pull
  ```

4. Ejecutar el pipeline completo de datos de forma reproducible:
  ```bash
  dvc repro
  dvc dag
  ```

5. Ejecutar la batería de pruebas automatizadas de calidad de datos:
  ```bash
  pytest --html=reports/reporte_calidad.html --self-contained-html
  ```