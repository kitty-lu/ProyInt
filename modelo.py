# modelo.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# imbalanced-learn
from imblearn.over_sampling import SMOTE

# Columnas esperadas (según tu script)
EXPECTED_COLUMNS = [
    'restecg','cigsPerDay','currentSmoker','heartRate','thal','prevalentHyp','age',
    'prevalentStroke','id','dataset','exang','BPMeds','diabetes','oldpeak','ca',
    'source','cp','sysBP','BMI','thalch','slope','glucose','education','chol','diaBP','target','sex'
]

def cargar_datos(path):
    """Carga CSV y devuelve DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. Coloca el CSV en la carpeta del proyecto.")
    df = pd.read_csv(path)
    return df

def limpiar_y_preprocesar(df):
    """
    Limpieza y preprocesamiento:
    - Dropna en target
    - Relleno de numéricas por media, categóricas por moda
    - Dummies para categóricas definidas en el notebook
    - Drop columna id si existe
    - Retorna X (numéricas) y y
    """
    df = df.copy()
    # Asegurarse de que target exista
    if 'target' not in df.columns:
        raise KeyError("La columna 'target' no existe en el dataset.")

    # Columnas categóricas que usaste en el script original
    categorical_cols = ['restecg', 'thal', 'dataset', 'exang', 'cp', 'slope', 'source', 'sex']
    # Columnas numéricas (según tu script)
    numerical_cols = ['cigsPerDay', 'currentSmoker', 'heartRate', 'prevalentHyp', 'age',
                      'prevalentStroke', 'BPMeds', 'diabetes', 'oldpeak', 'ca',
                      'sysBP', 'BMI', 'thalch', 'glucose', 'education', 'chol', 'diaBP']

    # Drop filas sin target
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)

    # Relleno
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Dummies para categóricas (si existen)
    cols_to_dummy = [c for c in categorical_cols if c in df.columns]
    if cols_to_dummy:
        df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)

    # Drop id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Sólo columnas numéricas para X (ya con dummies)
    X = df.drop(columns=['target'])
    X = X.select_dtypes(include=[np.number])
    y = df['target']

    feature_names = X.columns.tolist()

    return X, y, feature_names

def escalar(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler

def entrenar_arbol(X, y, test_size=0.2, random_state=42, max_depth=5, use_smote=False):
    """
    Entrena DecisionTree. Si use_smote=True aplica SMOTE en entrenamiento.
    Retorna diccionario con modelo, métricas y objetos auxiliares.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=random_state)

    # Escalado
    X_train_scaled, X_test_scaled, scaler = escalar(X_train, X_test)

    # SMOTE opcional
    if use_smote:
        sm = SMOTE(random_state=random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    # Entrenamiento
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train_res, y_train_res)

    # Predicción
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy,
        'feature_names': X.columns.tolist()
    }

def entrenar_random_forest(X, y, use_smote=False, random_state=42, n_estimators=100, max_depth=5):
    """
    Entrena RandomForest (se usa después de SMOTE si se pasa).
    Retorna diccionario similar a entrenar_arbol.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=random_state)
    X_train_scaled, X_test_scaled, scaler = escalar(X_train, X_test)

    if use_smote:
        sm = SMOTE(random_state=random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=random_state, class_weight='balanced')
    rf.fit(X_train_res, y_train_res)
    y_pred = rf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': rf,
        'scaler': scaler,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy,
        'feature_names': X.columns.tolist(),
        'importances': getattr(rf, "feature_importances_", None)
    }

def exportar_arbol(model, feature_names, output_path="images/arbol.png", class_names=None, max_depth=3):
    """
    Guarda imagen del árbol (plot_tree) en output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(22, 12))
    plot_tree(model, feature_names=feature_names,
              class_names=class_names or ['0', '1'],
              filled=True, rounded=True, max_depth=max_depth, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    return output_path
