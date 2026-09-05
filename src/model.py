"""
model.py
---------
Treinamento e avaliação de modelos de Machine Learning (scikit-learn) para
prever a evasão escolar (classificação binária: evadiu = 0 ou 1).

Comparamos dois modelos simples e interpretáveis:
- Regressão Logística (baseline clássico para classificação binária)
- Random Forest (modelo de árvores, captura relações não lineares)

E avaliamos com métricas apropriadas para classificação, além de olhar
quais variáveis mais pesam na decisão do modelo (feature importance).
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("outputs")
MODEL_PATH = Path("outputs/modelo_evasao.joblib")

COLUNAS_FEATURES = [
    "idade",
    "semestre",
    "horas_estudo_semanal",
    "frequencia_percentual",
    "nota_media",
    "trabalha",
    "bolsa_estudo",
    "possui_dispositivo_proprio",
    "apoio_familiar",
    "indicador_risco_manual",
]


def preparar_dados_treino(df: pd.DataFrame):
    """Separa features (X) e alvo (y), depois divide em treino e teste."""
    df_modelo = df.copy()

    # Junta as colunas numéricas "manuais" com as dummies de categorias criadas
    # em data_processing.codificar_categoricas (se existirem no df).
    colunas_dummies = [c for c in df_modelo.columns if c.startswith("qualidade_internet_") or c.startswith("faixa_frequencia_")]
    features = COLUNAS_FEATURES + colunas_dummies
    features = [c for c in features if c in df_modelo.columns]

    X = df_modelo[features]
    y = df_modelo["evadiu"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, features


def treinar_modelos(X_train, y_train, X_test):
    """Treina Regressão Logística e Random Forest, retorna previsões de ambos."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo_logistico = LogisticRegression(max_iter=1000, random_state=42)
    modelo_logistico.fit(X_train_scaled, y_train)

    modelo_floresta = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42
    )
    modelo_floresta.fit(X_train, y_train)  # árvores não precisam de escala

    resultados = {
        "Regressão Logística": {
            "modelo": modelo_logistico,
            "previsoes": modelo_logistico.predict(X_test_scaled),
            "probabilidades": modelo_logistico.predict_proba(X_test_scaled)[:, 1],
        },
        "Random Forest": {
            "modelo": modelo_floresta,
            "previsoes": modelo_floresta.predict(X_test),
            "probabilidades": modelo_floresta.predict_proba(X_test)[:, 1],
        },
    }
    return resultados, scaler


def avaliar_modelo(nome: str, y_test, previsoes, probabilidades) -> dict:
    return {
        "modelo": nome,
        "acuracia": round(accuracy_score(y_test, previsoes), 3),
        "f1_score": round(f1_score(y_test, previsoes), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilidades), 3),
        "relatorio_detalhado": classification_report(
            y_test, previsoes, target_names=["Permaneceu", "Evadiu"], zero_division=0
        ),
    }


def plotar_matriz_confusao(nome: str, y_test, previsoes) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, previsoes, display_labels=["Permaneceu", "Evadiu"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Matriz de confusão - {nome}")
    caminho = OUTPUT_DIR / f"matriz_confusao_{nome.replace(' ', '_').lower()}.png"
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em {caminho}")


def plotar_curva_roc(resultados: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nome, dados in resultados.items():
        RocCurveDisplay.from_predictions(y_test, dados["probabilidades"], name=nome, ax=ax)
    ax.set_title("Curva ROC - comparação entre modelos")
    caminho = OUTPUT_DIR / "curva_roc_comparacao.png"
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em {caminho}")


def plotar_importancia_features(modelo_floresta, features: list[str]) -> None:
    importancias = pd.Series(modelo_floresta.feature_importances_, index=features)
    importancias = importancias.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    importancias.plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_title("Importância das variáveis (Random Forest)")
    ax.set_xlabel("Importância relativa")
    caminho = OUTPUT_DIR / "importancia_features.png"
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em {caminho}")


def rodar_pipeline_modelo(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, features = preparar_dados_treino(df)
    resultados, scaler = treinar_modelos(X_train, y_train, X_test)

    print("=== Avaliação dos modelos ===\n")
    for nome, dados in resultados.items():
        metrica = avaliar_modelo(nome, y_test, dados["previsoes"], dados["probabilidades"])
        print(f"--- {nome} ---")
        print(f"Acurácia: {metrica['acuracia']}  |  F1-score: {metrica['f1_score']}  |  ROC-AUC: {metrica['roc_auc']}")
        print(metrica["relatorio_detalhado"])
        plotar_matriz_confusao(nome, y_test, dados["previsoes"])

    plotar_curva_roc(resultados, y_test)
    plotar_importancia_features(resultados["Random Forest"]["modelo"], features)

    # Salva o modelo de melhor desempenho (Random Forest, neste caso) para reuso futuro
    joblib.dump(resultados["Random Forest"]["modelo"], MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    from data_processing import codificar_categoricas, pipeline_limpeza
    from generate_data import gerar_dataset

    df = gerar_dataset()
    df = pipeline_limpeza(df)
    df = codificar_categoricas(df)
    rodar_pipeline_modelo(df)
