"""
analysis.py
------------
Análise exploratória de dados (EDA) e testes estatísticos com scipy.

Gera gráficos (salvos em outputs/) e imprime resultados de testes de
hipótese, conectando teoria de Probabilidade e Estatística com um problema
real de negócio (evasão escolar).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera imagens sem precisar de tela (headless)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

OUTPUT_DIR = Path("outputs")
sns.set_theme(style="whitegrid")


def _salvar_figura(fig, nome_arquivo: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    caminho = OUTPUT_DIR / nome_arquivo
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em {caminho}")


def resumo_estatistico(df: pd.DataFrame) -> pd.DataFrame:
    """Estatística descritiva básica (média, desvio padrão, quartis etc.)."""
    return df.describe(include="all").transpose()


def grafico_distribuicao_evasao(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    contagem = df["evadiu"].value_counts().sort_index()
    sns.barplot(x=["Permaneceu", "Evadiu"], y=contagem.values, ax=ax, hue=["Permaneceu", "Evadiu"], legend=False)
    ax.set_title("Distribuição de alunos: evasão vs. permanência")
    ax.set_ylabel("Quantidade de alunos")
    _salvar_figura(fig, "distribuicao_evasao.png")


def grafico_correlacao(df: pd.DataFrame) -> None:
    colunas_numericas = df.select_dtypes(include="number").drop(columns=["aluno_id"], errors="ignore")
    corr = colunas_numericas.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlação entre variáveis numéricas")
    _salvar_figura(fig, "matriz_correlacao.png")


def grafico_frequencia_por_evasao(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="evadiu", y="frequencia_percentual", ax=ax, hue="evadiu", legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Permaneceu", "Evadiu"])
    ax.set_title("Frequência nas atividades por status de evasão")
    _salvar_figura(fig, "frequencia_por_evasao.png")


def teste_hipotese_frequencia(df: pd.DataFrame) -> dict:
    """Teste t de Student: a frequência média difere entre quem evadiu e quem não evadiu?

    H0: as médias de frequência são iguais entre os dois grupos.
    H1: as médias são diferentes.
    """
    grupo_evadiu = df.loc[df["evadiu"] == 1, "frequencia_percentual"]
    grupo_permaneceu = df.loc[df["evadiu"] == 0, "frequencia_percentual"]

    estatistica, p_valor = stats.ttest_ind(grupo_evadiu, grupo_permaneceu, equal_var=False)

    return {
        "teste": "t de Student (Welch)",
        "estatistica_t": round(float(estatistica), 3),
        "p_valor": round(float(p_valor), 5),
        "significativo_5pct": bool(p_valor < 0.05),
        "media_evadiu": round(float(grupo_evadiu.mean()), 2),
        "media_permaneceu": round(float(grupo_permaneceu.mean()), 2),
    }


def teste_qui_quadrado_internet(df: pd.DataFrame) -> dict:
    """Teste qui-quadrado: existe associação entre qualidade da internet e evasão?

    H0: as duas variáveis categóricas são independentes.
    H1: existe associação entre elas.
    """
    tabela_contingencia = pd.crosstab(df["qualidade_internet"], df["evadiu"])
    chi2, p_valor, graus_liberdade, _ = stats.chi2_contingency(tabela_contingencia)

    return {
        "teste": "Qui-quadrado de independência",
        "chi2": round(float(chi2), 3),
        "graus_liberdade": int(graus_liberdade),
        "p_valor": round(float(p_valor), 5),
        "significativo_5pct": bool(p_valor < 0.05),
    }


def rodar_analise_completa(df: pd.DataFrame) -> None:
    print("=== Resumo estatístico ===")
    print(resumo_estatistico(df)[["mean", "std", "min", "max"]].dropna())

    grafico_distribuicao_evasao(df)
    grafico_correlacao(df)
    grafico_frequencia_por_evasao(df)

    print("\n=== Teste de hipótese: frequência x evasão ===")
    print(teste_hipotese_frequencia(df))

    print("\n=== Teste de hipótese: qualidade da internet x evasão ===")
    print(teste_qui_quadrado_internet(df))


if __name__ == "__main__":
    from data_processing import pipeline_limpeza
    from generate_data import gerar_dataset

    df = pipeline_limpeza(gerar_dataset())
    rodar_analise_completa(df)
