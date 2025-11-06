import os
import pandas as pd
import gradio as gr
import mlflow
import mlflow.sklearn
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configurar Gemini API (Gemini-2.5-flash)
configure(api_key=GOOGLE_API_KEY)
gemini_model = GenerativeModel("gemini-2.5-flash")
mlflow.set_tracking_uri("file:///C:/Users/Christian/Documents/proyecto final mlops/project/mlruns")

# Cargar el modelo production desde MLflow Registry
MODEL_NAME = "wine-quality-model"
STAGE = "Production"
model_uri = f"models:/{MODEL_NAME}/{STAGE}"
model = mlflow.sklearn.load_model(model_uri)

def hacer_prediccion(input_dict):
    # Prepara los datos como DataFrame para scikit-learn
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    return pred

def prediccion_batch(csv_file):
    # Leer el archivo subido (espera pandas-readable CSV)
    df = pd.read_csv(csv_file)
    resultados = []
    explicaciones = []
    for _, row in df.iterrows():
        datos = row.to_dict()
        prediccion = model.predict(pd.DataFrame([datos]))[0]
        explicacion = generar_explicacion(datos, prediccion)
        resultados.append(prediccion)
        explicaciones.append(explicacion)
    # Crear un dataframe resultado
    df_result = df.copy()
    df_result['prediccion'] = resultados
    df_result['explicacion'] = explicaciones
    # Guardar como artifact el resultado completo
    with mlflow.start_run():
        df_result.to_csv("batch_resultados.csv", index=False)
        mlflow.log_artifact("batch_resultados.csv")
    return df_result


def generar_explicacion(input_dict, prediccion):
    # Convierte input a string para prompt
    caracteristicas = ", ".join([f"{k}: {v}" for k, v in input_dict.items()])
    prompt = (
        f"Predicción de calidad para vino blanco: {prediccion}. "
        f"Características químicas del vino: {caracteristicas}. "
        "Genera una explicación experta enológica de por qué se obtuvo esa calidad según las características."
    )
    # Gemini API
    respuesta = gemini_model.generate_content(prompt).text
    return respuesta

def gradio_fn(**inputs):
    with mlflow.start_run():
        prediccion = hacer_prediccion(inputs)
        explicacion = generar_explicacion(inputs, prediccion)
        # Guardar explicación como artifact
        mlflow.log_text(explicacion, "explicacion_genai.txt")
    return f"Calidad predicha (binaria): {prediccion}", explicacion

# Define los campos según las columnas químicas del dataset
example_input = {
    "fixed acidity": 7.0,
    "volatile acidity": 0.27,
    "citric acid": 0.36,
    "residual sugar": 20.7,
    "chlorides": 0.045,
    "free sulfur dioxide": 45.0,
    "total sulfur dioxide": 170.0,
    "density": 1.001,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8,
}

inputs_gr = [gr.Number(label=k, value=v) for k, v in example_input.items()]

def gradio_input_wrapper(*vals):
    if vals[-1] is not None:
        # procesamiento batch con CSV
        df_result = prediccion_batch(vals[-1])
        # Retorna ambos outputs: DataFrame para prediccion y también explicación (vacía o resumen)
        return df_result, "Batch predicción generada y guardada como artifact."
    else:
        # procesamiento manual
        inputs_dict = dict(zip(example_input.keys(), vals[:-1]))
        pred, explicacion = gradio_fn(**inputs_dict)
        return pred, explicacion


interface = gr.Interface(
    fn=gradio_input_wrapper,
    inputs=inputs_gr + [gr.File(label="Subir CSV")],
    outputs=[gr.Dataframe(label="Resultados batch"), gr.Textbox(label="Explicación profesional (GenAI)")],
    title="Predicción de calidad de vino blanco + Explicación IA"
)




if __name__ == "__main__":
    interface.launch()
