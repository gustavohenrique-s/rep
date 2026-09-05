"""
data_processing.py
-------------------
Limpeza, tratamento e engenharia de features com pandas e numpy.

Etapas clássicas de um pipeline de dados:
1. Checar e tratar valores faltantes.
2. Tratar outliers (usando a técnica do IQR - intervalo interquartil).
3. Criar novas features (feature engineering) que ajudam o modelo.
4. Codificar variáveis categóricas para uso em Machine Learning.
"""

import numpy as np
import pandas as pd


def relatorio_valores_faltantes(df: pd.DataFrame) -> pd.Series:
    """Retorna a contagem de valores nulos por coluna."""
    return df.isna().sum()[df.isna().sum() > 0]


def tratar_valores_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche valores faltantes numéricos com a mediana da coluna.

    Usamos a mediana (e não a média) porque ela é mais robusta a outliers.
    """
    df = df.copy()
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        if df[col].isna().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
    return df


def limitar_outliers_iqr(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Faz 'capping' de outliers usando a técnica do IQR (intervalo interquartil).

    Valores abaixo de Q1 - 1.5*IQR ou acima de Q3 + 1.5*IQR são limitados
    (clip) ao invés de removidos, para não perder linhas do dataset.
    """
    df = df.copy()
    for col in colunas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=limite_inferior, upper=limite_superior)
    return df


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas colunas derivadas (feature engineering)."""
    df = df.copy()

    # Categoriza frequência em faixas de risco
    df["faixa_frequencia"] = pd.cut(
        df["frequencia_percentual"],
        bins=[-1, 50, 75, 100],
        labels=["baixa", "media", "alta"],
    )

    # Indicador simples de "risco combinado" (regra de negócio, não é o modelo)
    df["indicador_risco_manual"] = (
        (df["frequencia_percentual"] < 60).astype(int)
        + (df["horas_estudo_semanal"] < 4).astype(int)
        + (df["qualidade_internet"] == "ruim").astype(int)
        + (df["possui_dispositivo_proprio"] == 0).astype(int)
    )

    return df


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma colunas categóricas (texto) em variáveis numéricas (one-hot encoding)."""
    colunas_categoricas = ["qualidade_internet", "faixa_frequencia"]
    return pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)


def pipeline_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    """Executa o pipeline completo de limpeza e engenharia de features."""
    df = tratar_valores_faltantes(df)
    df = limitar_outliers_iqr(df, ["horas_estudo_semanal", "nota_media"])
    df = criar_features(df)
    return df


if __name__ == "__main__":
    from generate_data import gerar_dataset

    df_bruto = gerar_dataset(n_alunos=200)
    print("Valores faltantes antes:\n", relatorio_valores_faltantes(df_bruto))

    df_limpo = pipeline_limpeza(df_bruto)
    print("\nValores faltantes depois:\n", relatorio_valores_faltantes(df_limpo))
    print("\nExemplo com novas features:\n", df_limpo.head())
