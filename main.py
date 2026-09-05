"""
main.py
--------
Ponto de entrada do projeto. Executa o pipeline completo, do início ao fim:

    1. Geração dos dados sintéticos dos alunos
    2. Armazenamento em banco de dados SQLite
    3. Limpeza e engenharia de features (pandas / numpy)
    4. Análise exploratória e testes estatísticos (scipy)
    5. Treinamento e avaliação de modelos de Machine Learning (scikit-learn)

Para rodar:
    python main.py
"""

from src import analysis, database, model
from src.data_processing import codificar_categoricas, pipeline_limpeza
from src.generate_data import gerar_dataset


def main() -> None:
    print("#" * 70)
    print("PREVISÃO DE EVASÃO ESCOLAR EM CURSOS EAD - PIPELINE COMPLETO")
    print("#" * 70)

    # 1) Geração dos dados
    print("\n[1/5] Gerando dataset sintético de alunos...")
    df_bruto = gerar_dataset(n_alunos=1200)
    print(f"-> {len(df_bruto)} registros gerados.")

    # 2) Persistência em SQLite
    print("\n[2/5] Salvando dados brutos no banco SQLite...")
    database.salvar_dataframe(df_bruto, "alunos_raw")
    print("-> Tabela 'alunos_raw' criada em data/evasao.db")

    # 3) Limpeza e engenharia de features
    print("\n[3/5] Limpando dados e criando novas features...")
    df_limpo = pipeline_limpeza(df_bruto)
    df_modelo = codificar_categoricas(df_limpo)
    database.salvar_dataframe(df_limpo, "alunos_tratados")
    print("-> Tabela 'alunos_tratados' salva no banco.")

    # 4) Análise exploratória e estatística
    print("\n[4/5] Rodando análise exploratória e testes estatísticos...")
    analysis.rodar_analise_completa(df_limpo)

    # 5) Machine Learning
    print("\n[5/5] Treinando e avaliando modelos preditivos...")
    model.rodar_pipeline_modelo(df_modelo)

    print("\n" + "=" * 70)
    print("Pipeline finalizado! Veja os gráficos gerados na pasta 'outputs/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
