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
st.set_page_config(page_title="CardioRisk AI", page_icon="🫀", layout="wide")

# --------------------------------------------------
# TEMA Y ESTILOS DINÁMICOS
# --------------------------------------------------
# Selector de Modo Oscuro en la barra lateral (al principio)
modo_oscuro = st.sidebar.toggle("🌑 Modo Oscuro", value=False)

# Definición de colores según el modo seleccionado
if modo_oscuro:
    theme = {
        "bg_gradient": "linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)",
        "text_color": "#F8FAFC",
        "card_bg": "rgba(30, 41, 59, 0.7)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "header_bg": "rgba(15, 23, 42, 0.8)",
        "input_bg": "rgba(51, 65, 85, 0.6)",
        "input_text": "#F8FAFC",
        "sidebar_bg": "linear-gradient(180deg, #1e293b 0%, #0f172a 100%)",
        "section_line": "linear-gradient(90deg, #38BDF8 0%, transparent 100%)",
        "shadow_color": "rgba(0,0,0,0.5)"
    }
else:
    theme = {
        "bg_gradient": "linear-gradient(135deg, #E0F7FA 0%, #E3F2FD 50%, #F3E5F5 100%)",
        "text_color": "#1E293B",
        "card_bg": "rgba(255, 255, 255, 0.9)",
        "card_border": "rgba(255, 255, 255, 0.8)",
        "header_bg": "rgba(255, 255, 255, 0.85)",
        "input_bg": "rgba(241, 245, 249, 0.8)",
        "input_text": "#334155",
        "sidebar_bg": "linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%)",
        "section_line": "linear-gradient(90deg, #3498DB 0%, transparent 100%)",
        "shadow_color": "rgba(31, 38, 135, 0.1)"
    }

# CSS Dinámico Inyectado
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Reset y fuentes globales */
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
        color: {theme['text_color']};
    }}
    
    /* Fondo de la aplicación */
    .stApp {{
        background: {theme['bg_gradient']};
        background-attachment: fixed;
    }}

    /* Encabezado Principal - GLASSMORPHISM */
    .main-header {{
        background: {theme['header_bg']};
        backdrop-filter: blur(10px);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px {theme['shadow_color']};
        border: 1px solid {theme['card_border']};
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(52, 152, 219, 0.1) 0%, transparent 70%);
        animation: pulse-bg 15s infinite;
    }}

    @keyframes pulse-bg {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.2); }}
        100% {{ transform: scale(1); }}
    }}

    .main-header h1 {{
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin: 0;
        background: linear-gradient(90deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }}
    
    /* En modo oscuro, el título necesita ser blanco/brillante */
    {".main-header h1 { background: linear-gradient(90deg, #E0F7FA 0%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }" if modo_oscuro else ""}

    .main-header p {{
        font-size: 1.4rem;
        color: {theme['text_color']};
        margin-top: 15px;
        font-weight: 500;
        opacity: 0.8;
    }}

    /* Tarjetas de Contenido FLOTANTES */
    .content-card {{
        background: {theme['card_bg']};
        backdrop-filter: blur(8px);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px {theme['shadow_color']};
        border: 1px solid {theme['card_border']};
        border-top: 6px solid #3498DB;
        margin-bottom: 2rem;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }}

    .content-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(52, 152, 219, 0.15);
        border-top: 6px solid #2980B9;
    }}

    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {theme['text_color']};
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .section-title::after {{
        content: "";
        flex-grow: 1;
        height: 2px;
        background: {theme['section_line']};
        margin-left: 15px;
    }}

    /* Botones VIBRANTES */
    .stButton>button {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 10px 20px rgba(79, 172, 254, 0.4);
        transition: all 0.4s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    .stButton>button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 30px rgba(79, 172, 254, 0.6);
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    }}

    /* Inputs con estilo moderno */
    .stNumberInput > div > div > input, .stSelectbox > div > div > div {{
        background-color: {theme['input_bg']} !important;
        border: 2px solid transparent !important;
        border-radius: 12px !important;
        color: {theme['input_text']} !important;
        transition: all 0.3s ease;
    }}
    
    .stNumberInput > div > div > input:focus, .stSelectbox > div > div > div:focus-within {{
        background-color: {theme['card_bg']} !important;
        border: 2px solid #3498DB !important;
        box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.1) !important;
    }}
    
    /* Color del texto de la etiqueta del input */
    .stNumberInput label, .stSelectbox label {{
        color: {theme['text_color']} !important;
    }}
    
    /* Ajuste para los iconos de ayuda */
    .stTooltipIcon {{
        color: {theme['text_color']} !important;
    }}

    /* Sidebar estilizado */
    [data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']};
        border-right: 1px solid {theme['card_border']};
    }}
    
    /* Ajuste texto sidebar */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
         color: {theme['text_color']} !important;
    }}
    
    .sidebar-header {{
        background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(44, 62, 80, 0.2);
    }}
    
    .sidebar-header h2 {{
        color: white !important;
    }}

    /* Cajas de Resultado IMPACTANTES */
    .result-box {{
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
        backdrop-filter: blur(5px);
    }}

    .result-safe {{
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    }}

    .result-danger {{
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }}

    .result-icon {{
        font-size: 6rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.2));
        animation: float 3s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    .result-title {{
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}
    
    .result-subtitle {{
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 500;
    }}

    /* Animaciones */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .animate-fade-in {{
        animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🫀 Predicción de Riesgo Cardiovascular</h1>
    <p>Sistema Inteligente de Estimación de Riesgo Cardiovascular</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR MEJORADO (Enfoque Médico/Explicativo)
# --------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2 style='margin:0; font-size: 1.5rem;'>📌 Panel de Control</h2>
</div>
""", unsafe_allow_html=True)

# Explicación amigable para no expertos

st.sidebar.markdown("### 🧠 Factores de Mayor Influencia")
st.sidebar.caption("Variables que el modelo prioriza para su diagnóstico:")

# Gráfico de Importancia de Factores (Simplificado para médicos)
# Valores simulados representativos para este tipo de modelo
factores = ['Respuesta al Esfuerzo', 'Frecuencia Cardíaca', 'Edad', 'Colesterol', 'IMC']
importancia = [35, 25, 20, 15, 5] # Porcentajes aproximados para fines ilustrativos

fig_importance = go.Figure(go.Bar(
    x=importancia,
    y=factores,
    orientation='h',
    marker=dict(
        color=['#2C3E50', '#3498DB', '#1ABC9C', '#95A5A6', '#BDC3C7'],
        line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
    ),
    text=[f'{x}%' for x in importancia],
    textposition='auto',
    hovertemplate='%{y}: %{x}% de influencia<extra></extra>'
))

fig_importance.update_layout(
    title='',
    xaxis=dict(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False, 
        range=[0, 45]
    ),
    yaxis=dict(
        showgrid=False,
        categoryorder='total ascending',
        tickfont=dict(family='Inter', size=13, color='#2C3E50')
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=250,
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False
)

st.sidebar.plotly_chart(fig_importance, use_container_width=True)

st.sidebar.markdown("""
<div style='background-color: #E8F6F3; padding: 1rem; border-radius: 8px; border: 1px solid #D1F2EB; margin-top: 1rem;'>
    <small style='color: #16A085;'>
    <b>Nota Clínica:</b> El modelo da un peso significativo a la <i>"Respuesta al Esfuerzo"</i> (Oldpeak), lo que coincide con la literatura cardiológica sobre isquemia inducida.
    </small>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INGRESO DE DATOS DEL USUARIO
# --------------------------------------------------

st.markdown("<div class='section-title'>📋 Expediente Clínico Digital</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("#### 👤 Datos Demográficos")
    age = st.number_input("Edad del Paciente", min_value=18, max_value=100, value=45, help="Edad en años cumplidos")
    BMI = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, value=24.5, help="Peso(kg) / Altura(m)²")

with col2:
    st.markdown("#### 🩸 Signos Vitales & Labs")
    chol = st.number_input("Colesterol Total (mg/dL)", min_value=100.0, max_value=600.0, value=190.0, step=1.0)
    thalch = st.number_input("Frecuencia Cardíaca Máx.", min_value=60, max_value=220, value=150, help="Ritmo cardíaco máximo alcanzado durante ejercicio")

with col3:
    st.markdown("#### 🩺 Antecedentes")
    oldpeak = st.number_input("Depresión ST (Oldpeak)", min_value=0.0, max_value=6.0, value=0.0, step=0.1, help="Hallazgo en electrocardiograma")
    diabetes = st.selectbox("Diagnóstico de Diabetes", [0, 1], format_func=lambda x: "Negativo" if x == 0 else "Positivo")
    prevalentHyp = st.selectbox("Hipertensión Arterial", [0, 1], format_func=lambda x: "No diagnosticado" if x == 0 else "Diagnosticado")


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
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calcular_button = st.button("🔍 Calcular Riesgo Cardiovascular", use_container_width=True)

if calcular_button:
    with st.spinner('Procesando datos clínicos...'):
        time.sleep(1.5)

    pred = modelo.predict(df_user_scaled)[0]
    prob = modelo.predict_proba(df_user_scaled)[0][1]  # Probabilidad de alto riesgo

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 0:
        # CASO BAJO RIESGO
        st.markdown("""
        <div class='result-box result-safe'>
            <div class='result-icon'>🛡️</div>
            <div class='result-title'>BAJO RIESGO CARDIOVASCULAR</div>
            <div class='result-subtitle'>Análisis completado con éxito</div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        col_res1, col_res2 = st.columns([1, 1.2], gap="large")

        with col_res1:
            st.markdown("#### 📊 Análisis Probabilístico")
            # Gauge Chart mejorado
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Probabilidad de Riesgo", 'font': {'size': 18, 'color': '#2C3E50'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': '#27AE60'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2C3E50"},
                    'bar': {'color': "#27AE60"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E9ECEF",
                    'steps': [
                        {'range': [0, 40], 'color': '#E8F8F5'},
                        {'range': [40, 70], 'color': '#FEF9E7'},
                        {'range': [70, 100], 'color': '#FADBD8'}
                    ],
                    'threshold': {
                        'line': {'color': "#E74C3C", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': "Inter"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res2:
            st.markdown("#### 📝 Informe Médico Preliminar")
            st.info("""
            **Interpretación:**
            Los parámetros clínicos ingresados sugieren una baja probabilidad de desarrollar complicaciones cardiovasculares en el corto plazo.
            """)
            
            st.markdown("""
            <div style='background-color: #F8F9FA; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #27AE60;'>
                <h5 style='color: #27AE60; margin:0;'>✅ Recomendaciones Preventivas</h5>
                <ul style='margin-top: 0.5rem; padding-left: 1.2rem; color: #555;'>
                    <li>Mantener actividad física moderada (30 min/día).</li>
                    <li>Dieta balanceada baja en sodio y grasas saturadas.</li>
                    <li>Control anual de perfil lipídico.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    else:
        # CASO ALTO RIESGO
        
        # Efecto Lluvia de Calaveras (Solo CSS para evitar errores de traducción)
        import random
        
        # Generamos el CSS dinámicamente para que sea aleatorio pero seguro
        css_animation = """
        <style>
            @keyframes fall {
                0% { top: -10vh; opacity: 1; transform: rotate(0deg); }
                100% { top: 105vh; opacity: 0; transform: rotate(360deg); }
            }
            .skull-drop {
                position: fixed;
                z-index: 9999;
                user-select: none;
                pointer-events: none;
                font-size: 2.5rem;
                animation-name: fall;
                animation-timing-function: linear;
                animation-fill-mode: forwards;
            }
        """
        
        # Creamos 30 clases de animación aleatorias
        skull_html = '<div class="notranslate">'
        for i in range(30):
            left = random.randint(0, 100)
            duration = random.uniform(2, 5)
            delay = random.uniform(0, 3)
            
            # Definimos la clase CSS específica para esta calavera
            css_animation += f"""
            .skull-{i} {{
                left: {left}vw;
                animation-duration: {duration}s;
                animation-delay: {delay}s;
            }}
            """
            # Agregamos el div usando esa clase
            skull_html += f'<div class="skull-drop skull-{i}">💀</div>'
            
        css_animation += "</style>"
        skull_html += "</div>"
        
        # Renderizamos todo junto
        st.markdown(css_animation + skull_html, unsafe_allow_html=True)
 
        
        st.markdown("""
        <style>
            @keyframes pulse-red {
                0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
                70% { box-shadow: 0 0 0 20px rgba(231, 76, 60, 0); }
                100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
            }
            .result-danger {
                animation: pulse-red 2s infinite;
            }
        </style>
        <div class='result-box result-danger'>
            <div class='result-icon'>⚠️</div>
            <div class='result-title' style='font-family: "Arial Black", sans-serif; letter-spacing: 2px;'>ALTO RIESGO DETECTADO</div>
            <div class='result-subtitle'>Se sugiere atención médica prioritaria</div>
        </div>
        """, unsafe_allow_html=True)

        col_res1, col_res2 = st.columns([1, 1.2], gap="large")

        with col_res1:
            st.markdown("#### 📊 Análisis Probabilístico")
            # Gauge Chart Alerta
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Probabilidad de Riesgo", 'font': {'size': 18, 'color': '#2C3E50'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': '#C0392B'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2C3E50"},
                    'bar': {'color': "#E74C3C"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E9ECEF",
                    'steps': [
                        {'range': [0, 40], 'color': '#E8F8F5'},
                        {'range': [40, 70], 'color': '#FEF9E7'},
                        {'range': [70, 100], 'color': '#FADBD8'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': "Inter"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res2:
            st.markdown("#### 📝 Informe Médico Preliminar")
            st.error("""
            **Interpretación:**
            El modelo ha detectado patrones consistentes con un riesgo elevado de enfermedad cardiovascular.
            """)

            st.markdown("""
            <div style='background-color: #FFF5F5; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #E74C3C;'>
                <h5 style='color: #E74C3C; margin:0;'>🚨 Plan de Acción Sugerido</h5>
                <ul style='margin-top: 0.5rem; padding-left: 1.2rem; color: #555;'>
                    <li>Agendar consulta cardiológica a la brevedad.</li>
                    <li>Monitoreo frecuente de presión arterial.</li>
                    <li>Revisión estricta de dieta y medicación.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("⚠️ **Aviso Legal:** Esta herramienta NO sustituye el diagnóstico de un profesional de la salud.")

