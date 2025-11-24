import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

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

# CSS personalizado para mejorar la estética
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        margin-top: 1rem;
        opacity: 0.95;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 0.8rem 3rem;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        animation: fadeIn 0.5s ease-in;
    }

    .danger-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(238, 9, 121, 0.3);
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>❤️ Predicción de Riesgo Cardiovascular</h1>
    <p>Aplicación interactiva basada en Machine Learning para estimar el nivel de riesgo cardiaco</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='color: white; margin: 0;'>📌 Información del Modelo</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.success("Modelo final entrenado con Random Forest + SMOTE.")

st.sidebar.subheader("📊 Métricas del Modelo")

# 👉 MODIFICA ESTOS VALORES CON LOS REALES DE TU NOTEBOOK
accuracy = 0.87
recall = 0.84
precision = 0.85

# Crear gráfico de métricas
fig_metrics = go.Figure()

fig_metrics.add_trace(go.Bar(
    x=['Accuracy', 'Recall', 'Precision'],
    y=[accuracy, recall, precision],
    marker=dict(
        color=['#667eea', '#11998e', '#ee0979'],
        line=dict(color='white', width=2)
    ),
    text=[f'{accuracy*100:.1f}%', f'{recall*100:.1f}%', f'{precision*100:.1f}%'],
    textposition='outside',
    textfont=dict(size=14, color='white', family='Arial Black')
))

fig_metrics.update_layout(
    title=dict(text='Rendimiento del Modelo', font=dict(size=16, color='white')),
    yaxis=dict(range=[0, 1], tickformat='.0%', gridcolor='rgba(255,255,255,0.2)', title='', showticklabels=False),
    xaxis=dict(title='', tickfont=dict(size=12, color='white')),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=300,
    margin=dict(t=50, b=20, l=20, r=20),
    showlegend=False
)

st.sidebar.plotly_chart(fig_metrics, use_container_width=True)
st.sidebar.caption("Estas métricas fueron calculadas en el conjunto de pruebas del modelo.")

# --------------------------------------------------
# INGRESO DE DATOS DEL USUARIO
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;'>
    <h2 style='color: white; margin: 0; font-size: 2rem;'>🧍 Ingrese sus datos clínicos</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='input-section'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Datos Personales")
    age = st.number_input("Edad (años)", min_value=18, max_value=100, value=40, help="Ingrese su edad actual")
    BMI = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, help="Índice de Masa Corporal")

with col2:
    st.markdown("### 🩸 Parámetros Clínicos")
    chol = st.number_input("Colesterol (mg/dL)", min_value=100.0, max_value=500.0, value=200.0, help="Nivel de colesterol en sangre")
    thalch = st.number_input("Frecuencia cardíaca máx (bpm)", min_value=60, max_value=220, value=150, help="Frecuencia cardíaca máxima alcanzada")

with col3:
    st.markdown("### 📋 Condiciones")
    oldpeak = st.number_input("Oldpeak (mm)", min_value=0.0, max_value=6.0, value=1.0, help="Depresión del segmento ST")
    diabetes = st.selectbox("¿Diabetes?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    prevalentHyp = st.selectbox("¿Hipertensión?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

st.markdown("</div>", unsafe_allow_html=True)

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
st.markdown("<br><br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calcular_button = st.button("🔍 Calcular Riesgo Cardiovascular", use_container_width=True)

if calcular_button:
    with st.spinner('Analizando sus datos...'):
        time.sleep(1)

    pred = modelo.predict(df_user_scaled)[0]
    prob = modelo.predict_proba(df_user_scaled)[0][1]  # Probabilidad de alto riesgo

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 0:
        st.markdown("""
        <div class='success-box'>
            <div style='font-size: 4rem;'>✅</div>
            <div>BAJO RIESGO CARDIOVASCULAR</div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Probabilidad de Riesgo Alto", 'font': {'size': 24}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                    'bar': {'color': "#11998e"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig_gauge.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=80, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "darkblue", 'family': "Arial"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 2rem; border-radius: 15px; margin-top: 2rem;'>
                <h3 style='color: #155724; text-align: center;'>📊 Interpretación del Resultado</h3>
                <p style='color: #155724; font-size: 1.1rem; text-align: center;'>
                    Sus parámetros clínicos indican un <strong>bajo riesgo</strong> de enfermedad cardiovascular.
                </p>
                <p style='color: #155724; font-size: 1.1rem; text-align: center;'>
                    <strong>Probabilidad de riesgo alto: {prob*100:.1f}%</strong>
                </p>
                <div style='text-align: center; margin-top: 1rem;'>
                    <div style='font-size: 3rem;'>💚</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='danger-box'>
            <div style='font-size: 4rem;'>⚠️</div>
            <div>ALTO RIESGO CARDIOVASCULAR</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Probabilidad de Riesgo Alto", 'font': {'size': 24}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkred"},
                    'bar': {'color': "#ee0979"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig_gauge.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=80, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "darkred", 'family': "Arial"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 2rem; border-radius: 15px; margin-top: 2rem;'>
                <h3 style='color: #721c24; text-align: center;'>📊 Interpretación del Resultado</h3>
                <p style='color: #721c24; font-size: 1.1rem; text-align: center;'>
                    Sus parámetros clínicos indican un <strong>alto riesgo</strong> de enfermedad cardiovascular.
                </p>
                <p style='color: #721c24; font-size: 1.1rem; text-align: center;'>
                    <strong>Probabilidad de riesgo alto: {prob*100:.1f}%</strong>
                </p>
                <div style='text-align: center; margin-top: 1rem;'>
                    <div style='font-size: 3rem;'>🚨</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("⚠️ **Importante:** Este resultado es informativo y no sustituye un diagnóstico clínico profesional. Consulte con su médico para una evaluación completa.")

    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-top: 2rem; text-align: center;'>
        <h4 style='color: white; margin: 0;'>💡 Recomendaciones Generales</h4>
        <p style='color: white; margin-top: 1rem; font-size: 1rem;'>
            Mantenga un estilo de vida saludable: ejercicio regular, dieta balanceada, control de peso y chequeos médicos periódicos.
        </p>
    </div>
    """, unsafe_allow_html=True)
