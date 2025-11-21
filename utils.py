# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm, labels=None):
    """
    Devuelve figura matplotlib con matriz de confusión.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    plt.tight_layout()
    return fig

def mostrar_importancias(importances, feature_names, top_n=10):
    """
    Devuelve figura matplotlib con top_n importancias.
    importances puede ser array o pandas Series
    """
    if importances is None:
        return None
    if not isinstance(importances, pd.Series):
        importances = pd.Series(importances, index=feature_names)
    imp_sorted = importances.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp_sorted.values, y=imp_sorted.index, ax=ax)
    ax.set_title(f"Top {top_n} Importancias de Variables")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig
