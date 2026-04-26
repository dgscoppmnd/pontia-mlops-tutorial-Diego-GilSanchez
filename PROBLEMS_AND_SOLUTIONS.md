# Documentación de Problemas y Soluciones 

## Introducción

Este documento registra los problemas técnicos encontrados durante el desarrollo del proyecto MLOps y las soluciones implementadas. Esto facilita la comprensión de decisiones de diseño y evita que futuros desarrolladores incurran en los mismos errores.

---

## Problemas Encontrados y Soluciones

### 1. Descarga del Dataset en GitHub Actions

**Problema:**
El dataset "Adult Income" necesitaba descargarse automáticamente en el pipeline de Build, pero los URLs del UCI Repository cambiaban o fallaban ocasionalmente.

**Síntomas:**
- Build pipeline fallaba con error de conexión al descargar el dataset
- El modelo no se entrenaba porque faltaban los datos de entrada

**Solución Implementada:**
```yaml
# En build.yml
- name: Download Adult dataset
  run: |
    mkdir -p data/raw
    curl -o data/raw/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    curl -o data/raw/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

- Se configuró curl con reintentos implícitos
- Se documentó el URL exacto del dataset en el README
- Se agregó el dataset a `.gitignore` para no ocupar espacio en el repo (es generado dinámicamente)

**Aprendizaje:**
Usar URLs estables y documentar la fuente del dataset. Considerar almacenar el dataset en un bucket de cloud si es crítico.

---

### 2. Variables de Entorno en Render

**Problema:**
El deploy workflow necesitaba conectar con Render, pero las credenciales no estaban configuradas correctamente.

**Síntomas:**
- Deploy pipeline completaba sin errores, pero no ejecutaba el despliegue
- El webhook `RENDER_DEPLOY_HOOK` no era invocado

**Solución Implementada:**
```yaml
# En deploy.yml
- name: Deploy to Render
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK }}
```

**Pasos seguidos:**
1. En GitHub: Settings → Secrets and variables → Actions → New repository secret
2. Agregado: `RENDER_DEPLOY_HOOK` con la URL del webhook de Render
3. En Render: Settings del servicio → copiar el Manual Deploy Hook

**Aprendizaje:**
Los secrets deben estar disponibles en el contexto del workflow. Usar `${{ secrets.VARIABLE }}` es la forma correcta y segura.

---

### 3. Estructura de Directorios para Deployment

**Problema:**
La API FastAPI estaba en la raíz del proyecto, pero Render esperaba los archivos de Python en un subdirectorio específico.

**Síntomas:**
- Render intentaba ejecutar `uvicorn` desde la raíz, pero no encontraba el módulo `app`
- Error: `ModuleNotFoundError: No module named 'app'`

**Solución Implementada:**
Se reorganizó la estructura así:
```
├── deployment/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── main.py
├── render.yml
└── requirements.txt
```

**En `render.yml` y en manual deploy:**
```yaml
buildCommand: "cd deployment && pip install -r requirements.txt"
startCommand: "cd deployment && uvicorn app.main:app --host 0.0.0.0 --port 8080"
```

**Aprendizaje:**
Separar el código de deployment (API) del código de entrenamiento facilita el mantenimiento y permite desplegar solo lo necesario en producción.

---

### 4. Versionado de Modelos y Releases

**Problema:**
Los artifacts del modelo (model.pkl, scaler.pkl, encoders.pkl) no estaban siendo versionados correctamente en GitHub Releases.

**Síntomas:**
- Cada build creaba archivos .pkl localmente pero no se guardaban en el repositorio
- No había forma de rastrear qué versión de modelo estaba en producción
- Los rollbacks eran imposibles sin acceso a builds anteriores

**Solución Implementada:**
En `build.yml`, se agregó el paso de Upload Model to GitHub Release:
```yaml
- name: Upload model to GitHub Release
  uses: softprops/action-gh-release@v1
  with:
    files: |
      models/model.pkl
      models/scaler.pkl
      models/encoders.pkl
    tag_name: model-${{ github.run_number }}
    body: "Model trained on ${{ github.event.head_commit.timestamp }}"
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Ventajas:**
- Cada build genera un release automáticamente con un número único
- Los artifacts quedan guardados y disponibles para descargar
- Facilita rollbacks rápidos

**Aprendizaje:**
Automatizar el versionado reduce errores manuales y proporciona trazabilidad completa.

---

### 5. Tests Unitarios vs Tests de Integración

**Problema:**
Inicialmente no estaba clara la diferencia entre tests unitarios e integración, y los tests fallaban por dependencias.

**Síntomas:**
- Tests unitarios importaban módulos del dataset (dependencia externa)
- Tests de integración no validaban el modelo entrenado correctamente
- Cobertura de código era baja y engañosa

**Solución Implementada:**

**Tests Unitarios** (`unit_tests/`):
- Prueban funciones aisladas sin dependencias externas
- Usan mocks para simular data loading
- Se ejecutan rápido en CI/CD

```python
# unit_tests/test_data_loader.py
def test_load_data_with_missing_values():
    # Mock data, sin descargar dataset real
    data = pd.DataFrame({...})
    result = clean_data(data)
    assert result.isnull().sum().sum() == 0
```

**Tests de Integración** (`model_tests/`):
- Prueban el flujo completo: carga data → entrena modelo → evalúa
- Usan dataset real
- Se ejecutan después del build
- Validan accuracy, precision, recall

**Aprendizaje:**
Una buena estrategia de testing tiene múltiples capas. Tests rápidos en CI, tests comprensivos en CD.

---

### 6. Configuración de Paths en GitHub Actions

**Problema:**
El script `src/main.py` usaba paths relativos que funcionaban localmente pero fallaban en el runner de GitHub Actions.

**Síntomas:**
- Build fallaba con `FileNotFoundError: data/raw/adult.data`
- Los tests pasaban localmente pero no en CI

**Solución Implementada:**
Se utilizó `os.path.join()` con paths absolutos o relativos a la raíz del repo:

```python
# src/main.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

def load_data():
    train_file = os.path.join(DATA_DIR, 'adult.data')
    test_file = os.path.join(DATA_DIR, 'adult.test')
    # ...
```

También se actualizó el runner para ejecutar desde la raíz:
```yaml
- name: Train and save model
  run: |
    cd src
    python main.py
```

**Aprendizaje:**
Usar paths relativos desde la raíz del proyecto evita confusiones. En GitHub Actions, el working directory es la raíz del repo.

---

### 7. Infrastructure as Code (IaC) con `render.yml`

**Problema:**
Configurar Render manualmente a través de la UI era propenso a errores y no estaba documentado.

**Síntomas:**
- Diferentes desarrolladores configuraban cosas ligeramente diferentes
- No había registro de cambios de configuración
- Reproducir el despliegue en otro entorno era complicado

**Solución Implementada:**
Se creó `render.yml` en la raíz del repositorio con toda la configuración:

```yaml
services:
  - type: web
    name: pontia-mlops-api
    env: python
    plan: free
    region: frankfurt
    buildCommand: "cd deployment && pip install -r requirements.txt"
    startCommand: "cd deployment && uvicorn app.main:app --host 0.0.0.0 --port 8080"
    envVars:
      - key: GITHUB_REPO
        value: {github_user}/{github_proyect}
```

Esto permite:
- Version control de la infraestructura
- Reproducibilidad
- Colaboración clara
- Code review de cambios de infraestructura

**Aprendizaje:**
IaC es una best practice en DevOps. Aplicarla incluso en proyectos pequeños enseña buenos hábitos.

---

### 8. Cobertura de Tests en GitHub Actions

**Problema:**
Los tests se ejecutaban en CI pero la cobertura no se reportaba correctamente en los PRs.

**Síntomas:**
- Los committers no veían qué tests se ejecutaron
- No había feedback visual sobre cobertura de código
- Era difícil identificar dónde estaban los gaps de testing

**Solución Implementada:**
En `integration.yml`, se agregó comentario automático de resultados:

```yaml
- name: Comment test results and coverage on PR
  uses: actions/github-script@v7
  if: always()
  with:
    script: |
      const fs = require('fs');
      const results = fs.readFileSync('test-results/results.log', 'utf8');
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `### 🧪 Test Results\n\`\`\`\n${results}\n\`\`\``
      });
```

Ahora cada PR muestra automáticamente:
- Número de tests ejecutados
- Cobertura porcentual
- Failures (si aplica)

**Aprendizaje:**
Feedback visual en PRs mejora la cultura de code quality.

---

## Mejoras Futuras

1. **Monitoring en Producción**: Agregar logs y métricas en la API para detectar issues temprano
2. **Canary Deployments**: Implementar gradual rollout de nuevas versiones
3. **Model Registry Avanzado**: Integrar MLflow server en producción para versionado completo
4. **Database**: Agregar PostgreSQL para logging de predicciones y auditoría
5. **Alertas**: Configurar alertas en Render cuando el servicio falla o tiene latencia alta

---

## Conclusiones

Este proyecto demostró la importancia de:
- ✅ Automatizar desde el inicio (CI/CD)
- ✅ Documentar decisiones de arquitectura
- ✅ Usar Infrastructure as Code
- ✅ Tener múltiples capas de testing
- ✅ Mantener un historial claro en Git

