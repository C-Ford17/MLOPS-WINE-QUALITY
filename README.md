# Proyecto Final MLOps: Clasificación de Calidad de Vino Blanco con Explicaciones IA

## Descripción
Este proyecto implementa un pipeline completo de machine learning para predecir la calidad del vino blanco con un modelo Random Forest, gestionado y reproducido con MLflow, y una aplicación web interactiva con Gradio que incluye explicaciones automáticas generadas con Google Gemini (Gen AI).

---

## Estructura del Proyecto

- `/data`: Datos originales y ejemplos de CSV para batch.
- `/notebooks`: Exploración y experimentación en Jupyter notebooks.
- `/project`: Código de entrenamiento, MLproject, entorno Conda.
- `/gradio_app`: Aplicación web con Gradio para interacción y explicación.
- `.env`: Variables de entorno para claves API (no incluido en repo).
- `requirements.txt`: Dependencias Python.
- `README.md`: Documentación del proyecto.

---

## Instalación

Crear y activar entorno virtual:
python -m venv venv
source venv/bin/activate # Linux / Mac
.\venv\Scripts\activate # Windows


Instalar dependencias:
pip install -r requirements.txt


Configurar archivo `.env` con la clave API de Google Gemini:
GOOGLE_API_KEY=tu_clave_real


---

## Uso

### Entrenamiento y Experimentos

Ejecutar experimentos con MLflow Projects:
mlflow run ./project -P n_estimators=100 -P max_depth=5 -P class_weight=balanced


Acceder a UI de MLflow:
mlflow ui


Para visualizar métricas y modelos registrados.

### Aplicación Interactiva Gradio

Ejecutar la app:
python gradio_app/app.py


Abrir la app en `http://localhost:7860` y usar los formularios para predicción individual o cargar CSV para batch, junto con explicación generada por IA.

---

## Resultados y Evidencias

- Capturas de pantalla de MLflow con runs y modelo registrado.
- Capturas y video demostrativo de la app funcionando.
- Archivos CSV de entrada y artifacts con explicaciones.

---

## Consideraciones Éticas

Se discuten las implicaciones del uso de IA generativa para explicar modelos, los posibles sesgos y la responsabilidad en el despliegue.

---

## Futuras Mejoras

- Validación y manejo de errores en la app con mayor detalle.
- Expansión para múltiples tipos de vino y modelos.
- Integración de pipelines CI/CD para despliegue continuo.

---

## Autor

Christian - Colombia 6 Noviembre 2025





