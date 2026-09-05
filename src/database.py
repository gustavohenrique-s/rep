"""
database.py
------------
Camada de persistência com SQLite.

Aqui mostramos o uso do sqlite3 (biblioteca padrão do Python) junto com
pandas para salvar e consultar dados — algo bem comum em projetos de dados
pequenos/médios que não justificam um banco de dados completo como
PostgreSQL, mas ainda precisam de armazenamento estruturado e consultas SQL.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/evasao.db")


def criar_conexao(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Cria (se necessário) e retorna uma conexão com o banco SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def salvar_dataframe(df: pd.DataFrame, tabela: str, db_path: Path = DB_PATH) -> None:
    """Salva um DataFrame inteiro como uma tabela no SQLite (substitui se existir)."""
    conn = criar_conexao(db_path)
    try:
        df.to_sql(tabela, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def carregar_tabela(tabela: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Carrega uma tabela inteira do SQLite como DataFrame."""
    conn = criar_conexao(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {tabela}", conn)
    finally:
        conn.close()


def executar_query(query: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Executa uma query SQL arbitrária (SELECT) e retorna como DataFrame.

    Útil para mostrar em entrevista: "eu sei escrever SQL e também usar
    pandas para pegar o resultado já pronto para análise".
    """
    conn = criar_conexao(db_path)
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    # Pequeno teste manual do módulo
    from generate_data import gerar_dataset

    df = gerar_dataset(n_alunos=50)
    salvar_dataframe(df, "alunos")
    print("Tabela 'alunos' salva com sucesso.")

    resultado = executar_query(
        "SELECT qualidade_internet, AVG(evadiu) as taxa_evasao "
        "FROM alunos GROUP BY qualidade_internet"
    )
    print(resultado)
