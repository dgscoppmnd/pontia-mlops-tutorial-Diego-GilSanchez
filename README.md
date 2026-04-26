# PontIA MLOps Tutorial - Diego Gil Sanchez

Este repositorio es un tutorial completo de MLOps (Machine Learning Operations) que demuestra el ciclo de vida de un modelo de machine learning, desde el entrenamiento hasta el despliegue en producción. El proyecto entrena un modelo de clasificación RandomForest para predecir si el ingreso anual de una persona supera los $50,000 basado en el dataset "Adult Income" del UCI Machine Learning Repository.

## Funcionalidad General

El proyecto implementa un pipeline de ML que incluye:
- **Carga y preprocesamiento de datos**: Limpieza, encoding de variables categóricas y escalado de features.
- **Entrenamiento del modelo**: Un clasificador RandomForest optimizado.
- **Evaluación**: Métricas de accuracy y reporte de clasificación.
- **Registro y versionado**: Uso de MLflow para tracking y registro de modelos.
- **Despliegue**: API REST con FastAPI desplegada en Render, que descarga automáticamente la última versión del modelo desde GitHub Releases.

## Flujo CI/CD

El proyecto utiliza tres workflows de GitHub Actions para cubrir las etapas principales del ciclo DevOps/MLOps:

### Integration

El workflow `integration.yml` valida la integración continua del proyecto. Instala las dependencias definidas en `requirements.txt` y ejecuta los tests necesarios para comprobar que los cambios no rompen la funcionalidad existente antes de integrarse en `main`.

### Build

El workflow `build.yml` se encarga de construir el artefacto del modelo. Descarga los datos necesarios, entrena el modelo, ejecuta tests de integración y rendimiento, y genera o registra el modelo para su posterior despliegue.

### Deploy

El workflow `deploy.yml` ejecuta el despliegue de la API en Render mediante el deploy hook configurado como secreto en GitHub. Esta etapa permite publicar el servicio una vez que el modelo y la aplicación han sido validados.


## Estructura de Directorios

```
.github/
└── workflows/
    ├── integration.yml         # Workflow de integración continua: instala dependencias y ejecuta tests
    ├── build.yml               # Workflow de build: entrena, valida y registra/genera el modelo
    └── deploy.yml              # Workflow de despliegue en Render
├── data/
│   ├── raw/                    # Datos crudos del dataset Adult Income
│   └── deployment/
│       └── requirements.txt    # Dependencias específicas para el despliegue
├── deployment/
│   └── app/
│       ├── __init__.py
│       └── main.py             # Aplicación FastAPI para servir predicciones
├── model_tests/                # Tests de integración del modelo
├── models/                     # Directorio para guardar modelos entrenados localmente
├── scripts/
│   └── register_model.py       # Script para registrar el modelo en MLflow
├── src/                        # Código fuente principal
│   ├── __init__.py
│   ├── data_loader.py          # Funciones para cargar y preprocesar datos
│   ├── evaluate.py             # Funciones de evaluación del modelo
│   ├── main.py                 # Script principal para entrenamiento
│   └── model.py                # Definición y entrenamiento del modelo
├── unit_tests/                 # Tests unitarios
├── pytest.ini                  # Configuración de pytest
├── render.yml                  # Configuración de despliegue para Render
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

### Descripción de Componentes Principales

- **`.github/workflows/deploy.yml`**: Automatiza el despliegue en Render mediante un webhook cuando se dispara manualmente.
- **`data/raw/`**: Contiene los archivos `adult.data` y `adult.test` del dataset Adult Income.
- **`deployment/app/main.py`**: API FastAPI que:
  - Descarga el modelo desde GitHub Releases al iniciar.
  - Expone endpoints para predicciones (`/predict`) y health check (`/health`).
  - Maneja métricas básicas de uso.
- **`scripts/register_model.py`**: Registra el modelo entrenado en MLflow, lo transita a "Staging" y lo marca como "champion".
- **`src/`**:
  - `data_loader.py`: Carga datos CSV, maneja valores faltantes, aplica label encoding y scaling.
  - `evaluate.py`: Calcula accuracy y genera reporte de clasificación.
  - `main.py`: Orquesta el entrenamiento completo, logging y guardado de artifacts.
  - `model.py`: Define y entrena el RandomForestClassifier.
- **`unit_tests/` y `model_tests/`**: Suites de tests para validar funcionalidad y rendimiento.

## Cómo Poner en Marcha el Proyecto

### Prerrequisitos

- Python 3.10+
- Git
- Cuenta en GitHub (para releases)
- Cuenta en Render (opcional para despliegue)
- MLflow server local (opcional para registro avanzado)

### Instalación y Configuración

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/dgscoppmnd/pontia-mlops-tutorial-Diego-GilSanchez.git
   cd pontia-mlops-tutorial-Diego-GilSanchez
   ```

2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Descarga los datos** (si no están incluidos):
   - El dataset Adult Income debe estar en `data/raw/adult.data` y `data/raw/adult.test`.
   - Puedes descargarlos desde [UCI Repository](https://archive.ics.uci.edu/dataset/2/adult).

### Ejecución Local

1. **Ejecuta los tests**:
   ```bash
   pytest unit_tests/ model_tests/
   ```

2. **Entrena el modelo**:
   ```bash
   python src/main.py
   ```
   Esto generará logs en `training.log` y guardará el modelo en `models/model.pkl`.

3. **Evalúa el modelo** (incluido en el paso anterior):
   - El script `main.py` ya incluye evaluación automática.

4. **Registra el modelo en MLflow** (opcional):
   - Asegúrate de que MLflow esté corriendo: `mlflow server --host 127.0.0.1 --port 5000`
   - Ejecuta: `python scripts/register_model.py`
   - Necesitas crear un archivo `run_id.txt` con el ID del run de MLflow.

### Despliegue

1. **Prepara el despliegue**:
   - Crea un release en GitHub con los artifacts del modelo (model.pkl, scaler.pkl, encoders.pkl).
   - Actualiza `render.yml` con tu repositorio de GitHub.

2. **Despliega en Render**:
   - Conecta tu repositorio a Render.
   - Usa la configuración en `render.yml`.
   - El workflow `.github/workflows/deploy.yml` puede disparar despliegues automáticos.

3. **Prueba la API**:
   - Una vez desplegada, la API estará disponible en la URL de Render.
   - Endpoint `/predict`: Envía datos JSON para predicciones.
   - Endpoint `/health`: Verifica el estado del servicio.

## Simulación de un Proceso de Rollback

En un entorno de producción, los rollbacks son necesarios cuando una nueva versión del modelo introduce problemas (como degradación de performance o errores). Este proyecto simula un rollback usando MLflow para versionado de modelos.

### Escenario de Rollback

Imagina que has desplegado la versión 2.0 del modelo, pero los usuarios reportan una disminución en la accuracy. Necesitas revertir rápidamente a la versión 1.0 anterior.

### Pasos para Simular el Rollback

1. **Identifica las versiones**:
   - En MLflow UI, revisa el modelo registrado (ej. "adult-income-model").
   - Versión 2.0 está marcada como "champion" (en producción).
   - Versión 1.0 está en "Archived" o "Staging".

2. **Cambia el alias en MLflow**:
   ```python
   from mlflow.tracking import MlflowClient

   client = MlflowClient()
   model_name = "adult-income-model"

   # Remueve el alias "champion" de la versión 2.0
   client.delete_registered_model_alias(model_name, "champion")

   # Asigna "champion" a la versión 1.0
   client.set_registered_model_alias(model_name, "champion", "1")
   ```

3. **Actualiza el despliegue**:
   - Si la app descarga automáticamente desde GitHub Releases, crea un nuevo release con la versión 1.0 del modelo.
   - Actualiza el tag en `render.yml` o variables de entorno para apuntar al release anterior.
   - Dispara el workflow de despliegue para redeployar con la versión anterior.

4. **Verifica el rollback**:
   - Monitorea las métricas de la API (`/health`).
   - Ejecuta tests de integración para confirmar que la versión anterior funciona correctamente.
   - Notifica a stakeholders sobre el rollback.

### Mejores Prácticas para Rollbacks

- Mantén múltiples versiones del modelo en staging.
- Implementa canary deployments para probar nuevas versiones con un subset de usuarios.
- Automatiza el proceso de rollback con scripts o pipelines CI/CD.
- Registra métricas de performance en producción para detectar issues temprano.

## Contribución

Si deseas contribuir:
1. Crea un fork del repositorio.
2. Crea una rama para tu feature.
3. Ejecuta los tests antes de hacer commit.
4. Abre un Pull Request con descripción detallada.

## Licencia

Este proyecto es para fines educativos. Consulta la licencia del dataset Adult Income para uso comercial.