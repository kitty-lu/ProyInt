# app.py
import streamlit as st
import os
import pandas as pd
from modelo import cargar_datos, limpiar_y_preprocesar, entrenar_arbol, entrenar_random_forest, exportar_arbol
from utils import plot_confusion_matrix, mostrar_importancias
import matplotlib.pyplot as plt

st.set_page_config(page_title="Riesgo Cardiovascular - Dashboard", layout="wide")

st.title("🩺 Dashboard: Riesgo Cardiovascular (Árbol de Decisión + SMOTE)")

# Ruta por defecto al CSV (ajusta si tu archivo tiene otro nombre)
DEFAULT_CSV = "heart_data_unificado (2).csv"

st.sidebar.header("Configuración")
csv_path = st.sidebar.text_input("Ruta al dataset CSV", value=DEFAULT_CSV)
use_smote = st.sidebar.checkbox("Aplicar SMOTE en entrenamiento", value=True)
train_rf = st.sidebar.checkbox("Entrenar Random Forest (adicional)", value=True)
max_depth = st.sidebar.slider("Máx. profundidad del árbol", min_value=2, max_value=12, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("Referencia del script original subido:")
st.sidebar.text("/mnt/data/arbol_de_riesgo_cardio_(1).py")

# Cargar dataset
if not os.path.exists(csv_path):
    st.error(f"No se encontró el archivo: {csv_path}")
    st.info("Coloca el CSV en la carpeta del proyecto o ajusta la ruta en la barra lateral.")
    st.stop()

with st.spinner("Cargando dataset..."):
    df = cargar_datos(csv_path)

st.subheader("Vista previa del dataset")
st.dataframe(df.head(8))

# Mostrar estadísticas clave si existen
if set(['age','BMI','chol','sysBP','thalch','target']).issubset(df.columns):
    st.write("Estadísticas (age, BMI, chol, sysBP, thalch, target):")
    st.table(df[['age','BMI','chol','sysBP','thalch','target']].describe().T)

# Preprocesamiento
st.subheader("Preprocesamiento")
X, y, feature_names = limpiar_y_preprocesar(df)
st.write(f"Variables finales (numéricas y dummies): {len(feature_names)} columnas")

# Entrenamiento
if st.button("Entrenar modelo (Decision Tree)"):
    with st.spinner("Entrenando..."):
        res = entrenar_arbol(X, y, max_depth=max_depth, use_smote=use_smote)
        model = res['model']
        cm = res['confusion_matrix']
        report = res['classification_report']
        acc = res['accuracy']
        feature_names = res['feature_names']
        st.success(f"Entrenamiento completado. Accuracy: {acc:.4f}")

        # Matriz de confusión
        fig_cm = plot_confusion_matrix(cm, labels=['Bajo (0)','Alto (1)'])
        st.pyplot(fig_cm)

        # Reporte
        st.subheader("Reporte de Clasificación (Decision Tree)")
        st.text(pd.DataFrame(report).transpose().to_string())

        # Exportar árbol e mostrar
        img_path = exportar_arbol(model, feature_names, output_path="images/arbol.png", class_names=['Bajo Riesgo (0)', 'Alto Riesgo (1)'], max_depth=3)
        st.image(img_path, caption="Árbol de Decisión (vista resumida)")

        # Importancias (si aplica)
        try:
            importances = model.feature_importances_
            fig_imp = mostrar_importancias(importances, feature_names, top_n=10)
            if fig_imp:
                st.pyplot(fig_imp)
        except Exception:
            st.info("No se pudo obtener importancias para el Decision Tree.")

        # Entrenar Random Forest opcional
        if train_rf:
            st.info("Entrenando Random Forest (esto puede tardar unos segundos)...")
            rf_res = entrenar_random_forest(X, y, use_smote=use_smote, max_depth=5)
            st.success(f"Random Forest accuracy: {rf_res['accuracy']:.4f}")
            st.subheader("Matriz de Confusión (Random Forest)")
            st.pyplot(plot_confusion_matrix(rf_res['confusion_matrix'], labels=['Bajo (0)','Alto (1)']))
            st.subheader("Top Importancias (Random Forest)")
            if rf_res['importances'] is not None:
                st.pyplot(mostrar_importancias(rf_res['importances'], rf_res['feature_names'], top_n=10))
            else:
                st.info("Random Forest no proporcionó importancias.")
