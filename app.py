import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --------------------------------------------------
# CARGA DEL MODELO Y SCALER
# --------------------------------------------------
modelo = joblib.load("modelo_cardio.joblib")      # Modelo RandomForest optimizado
scaler = joblib.load("scaler.joblib")             # Escalador usado en el entrenamiento
feature_names = joblib.load("feature_names.joblib")  # Lista de columnas originales

# --------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------
st.set_page_config(page_title="Predicción de Riesgo Cardiovascular", page_icon="❤️", layout="wide")

st.title("❤️ Predicción de Riesgo Cardiovascular")
st.write("""Aplicación interactiva basada en **Machine Learning** para estimar 
el nivel de riesgo cardiaco de un paciente según sus parámetros clínicos.""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📌 Información del Modelo")
st.sidebar.success("Modelo final entrenado con Random Forest + SMOTE.")

st.sidebar.subheader("📊 Métricas del Modelo")

# 👉 MODIFICA ESTOS VALORES CON LOS REALES DE TU NOTEBOOK
accuracy = 0.87
recall = 0.84
precision = 0.85

st.sidebar.write(f"**Accuracy:** {accuracy*100:.2f}%")
st.sidebar.write(f"**Recall:** {recall*100:.2f}%")
st.sidebar.write(f"**Precision:** {precision*100:.2f}%")
st.sidebar.write("---")
st.sidebar.caption("Estas métricas fueron calculadas en el conjunto de pruebas del modelo.")

# --------------------------------------------------
# INGRESO DE DATOS DEL USUARIO
# --------------------------------------------------
st.header("🧍 Ingrese sus datos clínicos")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad (años)", min_value=18, max_value=100, value=40)
    BMI = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    chol = st.number_input("Colesterol (mg/dL)", min_value=100.0, max_value=500.0, value=200.0)
    thalch = st.number_input("Frecuencia cardíaca máx (bpm)", min_value=60, max_value=220, value=150)

with col3:
    oldpeak = st.number_input("Oldpeak (mm)", min_value=0.0, max_value=6.0, value=1.0)
    diabetes = st.selectbox("¿Diabetes?", [0, 1])
    prevalentHyp = st.selectbox("¿Hipertensión?", [0, 1])

# Crear vector del usuario con los valores faltantes = 0
input_dict = {col: 0 for col in feature_names}

# Actualizar las variables que el usuario ingresa
input_dict["age"] = age
input_dict["BMI"] = BMI
input_dict["chol"] = chol
input_dict["thalch"] = thalch
input_dict["oldpeak"] = oldpeak
input_dict["diabetes"] = diabetes
input_dict["prevalentHyp"] = prevalentHyp

# Convertir en DataFrame
df_user = pd.DataFrame([input_dict], columns=feature_names)

# Estandarizar
df_user_scaled = scaler.transform(df_user)

# --------------------------------------------------
# RESULTADOS
# --------------------------------------------------
st.header("🩺 Resultado de la Predicción")

if st.button("Calcular Riesgo"):
    pred = modelo.predict(df_user_scaled)[0]
    prob = modelo.predict_proba(df_user_scaled)[0][1]  # Probabilidad de alto riesgo

    if pred == 0:
        st.success(f"🟢 **BAJO RIESGO**")
    else:
        st.error(f"🔴 **ALTO RIESGO**")

    st.subheader("📈 Probabilidad del Modelo")
    st.write(f"**Probabilidad estimada de Riesgo Alto:** {prob*100:.2f}%")

    st.info("⚠️ Este resultado es informativo y no sustituye un diagnóstico clínico profesional.")
