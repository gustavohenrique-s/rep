"""
generate_data.py
-----------------
Gera um dataset SINTÉTICO (fictício, mas com relações estatísticas realistas)
de alunos de um curso EAD, simulando fatores que a literatura de educação a
distância aponta como relevantes para a evasão escolar:

- frequência nas atividades
- horas de estudo semanais
- qualidade do acesso à internet
- possuir dispositivo próprio (notebook/PC) para estudar
- carga de trabalho (aluno que trabalha tem menos tempo disponível)
- apoio familiar percebido
- nota média nas avaliações
- bolsa de estudo

Por que dados sintéticos?
Como é um projeto de portfólio, evitamos usar dados reais de alunos (questão
de privacidade / LGPD). Geramos dados artificiais com o NumPy, mas
construídos para terem correlações plausíveis com a variável alvo (evasão),
o que também é uma boa prática discutir em entrevista: "eu sei simular dados
com relações estatísticas controladas para testar meu pipeline".
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def gerar_dataset(n_alunos: int = 1200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Gera um DataFrame com dados sintéticos de alunos de um curso EAD.

    Parameters
    ----------
    n_alunos : int
        Quantidade de alunos a simular.
    seed : int
        Semente para reprodutibilidade (mesmo resultado sempre que rodar).

    Returns
    -------
    pd.DataFrame
        Dataset com features dos alunos e a coluna alvo `evadiu`.
    """
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 45, size=n_alunos)
    semestre = rng.integers(1, 9, size=n_alunos)

    horas_estudo_semanal = np.clip(rng.normal(8, 4, n_alunos), 0, 30)
    frequencia_percentual = np.clip(rng.normal(75, 18, n_alunos), 0, 100)
    nota_media = np.clip(rng.normal(6.5, 1.8, n_alunos), 0, 10)

    trabalha = rng.choice([0, 1], size=n_alunos, p=[0.35, 0.65])
    bolsa_estudo = rng.choice([0, 1], size=n_alunos, p=[0.6, 0.4])
    possui_dispositivo_proprio = rng.choice([0, 1], size=n_alunos, p=[0.15, 0.85])

    qualidade_internet = rng.choice(
        ["ruim", "media", "boa"], size=n_alunos, p=[0.2, 0.45, 0.35]
    )
    apoio_familiar = rng.integers(1, 6, size=n_alunos)  # escala 1 (baixo) a 5 (alto)

    # Mapear categorias para valores numéricos usados só no cálculo da probabilidade
    # (centralizados em 0 para que a escala fique comparável às demais variáveis)
    internet_score = pd.Series(qualidade_internet).map(
        {"ruim": -0.5, "media": 0.0, "boa": 0.5}
    ).to_numpy()
    dispositivo_centrado = possui_dispositivo_proprio - 0.85

    # Padronização (z-score) das variáveis contínuas usando os parâmetros que
    # sabemos ter sido usados para gerá-las. Isso deixa cada coeficiente do
    # logit numa escala comparável (efeito por "desvio-padrão" da variável),
    # em vez de depender da escala bruta (0-100 vs 0-10 vs 0-30 etc.).
    z_horas = (horas_estudo_semanal - 8) / 4
    z_frequencia = (frequencia_percentual - 75) / 18
    z_nota = (nota_media - 6.5) / 1.8
    z_apoio = (apoio_familiar - 3) / 1.5

    # -----------------------------------------------------------------
    # Construção da probabilidade de evasão via regressão logística
    # "de mentira": escolhemos os pesos manualmente para simular relações
    # plausíveis (e com efeito forte o suficiente para aparecer nos testes
    # estatísticos e no modelo depois). É assim que, em muitos tutoriais e
    # provas de conceito, se cria um dataset sintético para classificação.
    # -----------------------------------------------------------------
    logit = (
        -1.4
        + 0.9 * trabalha
        - 0.7 * bolsa_estudo
        - 0.5 * z_horas
        - 0.6 * z_frequencia
        - 0.6 * z_nota
        - 1.1 * internet_score
        - 0.8 * dispositivo_centrado
        - 0.4 * z_apoio
        + rng.normal(0, 0.5, n_alunos)  # ruído aleatório (nem tudo é explicável)
    )
    prob_evasao = 1 / (1 + np.exp(-logit))
    evadiu = rng.binomial(1, prob_evasao)

    df = pd.DataFrame(
        {
            "aluno_id": np.arange(1, n_alunos + 1),
            "idade": idade,
            "semestre": semestre,
            "horas_estudo_semanal": horas_estudo_semanal.round(1),
            "frequencia_percentual": frequencia_percentual.round(1),
            "nota_media": nota_media.round(1),
            "trabalha": trabalha,
            "bolsa_estudo": bolsa_estudo,
            "possui_dispositivo_proprio": possui_dispositivo_proprio,
            "qualidade_internet": qualidade_internet,
            "apoio_familiar": apoio_familiar,
            "evadiu": evadiu,
        }
    )

    # Simular alguns dados faltantes/sujos de propósito, como acontece na vida real
    # (isso dá material real para mostrar limpeza de dados com pandas na entrevista)
    idx_faltantes = rng.choice(df.index, size=int(n_alunos * 0.03), replace=False)
    df.loc[idx_faltantes, "nota_media"] = np.nan

    return df


if __name__ == "__main__":
    dataset = gerar_dataset()
    caminho_csv = "data/alunos_raw.csv"
    dataset.to_csv(caminho_csv, index=False)
    print(f"Dataset gerado com {len(dataset)} alunos -> {caminho_csv}")
    print(dataset.head())
    print("\nTaxa de evasão simulada: {:.1%}".format(dataset["evadiu"].mean()))
