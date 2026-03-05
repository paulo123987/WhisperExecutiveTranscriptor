"""
Utilitário de conexão Oracle e geração de consulta de amostragem estatística.

Gera e (opcionalmente) executa a query que busca N registros únicos por CAMPO1,
aleatoriamente, para um mês específico (AAAA-MM), usando DBMS_RANDOM.VALUE do Oracle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import oracledb
    import pandas as pd

logger = logging.getLogger(__name__)

# ── SQL template ──────────────────────────────────────────────────────────────
_SAMPLE_SQL_TEMPLATE = """\
SELECT
    CAMPO1,
    CAMPO2,
    TO_CHAR(START_TIME, 'YYYY-MM-DD HH24:MI:SS') AS START_DATETIME,
    TRANSCRICAO
FROM (
    SELECT
        CAMPO1,
        CAMPO2,
        START_TIME,
        TRANSCRICAO,
        ROW_NUMBER() OVER (
            PARTITION BY CAMPO1
            ORDER BY DBMS_RANDOM.VALUE
        ) AS rn
    FROM {tabela}
    WHERE START_TIME >= TO_DATE('{ano_mes}-01', 'YYYY-MM-DD')
      AND START_TIME <  ADD_MONTHS(TO_DATE('{ano_mes}-01', 'YYYY-MM-DD'), 1)
      AND CAMPO1 IN ({campos_in})
)
WHERE rn = 1
ORDER BY DBMS_RANDOM.VALUE
FETCH FIRST {n_amostras} ROWS ONLY"""


def build_sample_query(
    ano_mes: str,
    campos: list[str],
    tabela: str = "TABELA",
    n_amostras: int = 6000,
) -> str:
    """
    Constrói a query Oracle de amostragem estatística aleatória.

    Parâmetros
    ----------
    ano_mes:    Período no formato 'AAAA-MM' (ex.: '2024-03').
    campos:     Lista de valores de CAMPO1 para filtrar (ex.: ['123', '1234']).
    tabela:     Nome da tabela Oracle (default: 'TABELA').
    n_amostras: Quantidade máxima de registros únicos por CAMPO1 a retornar
                (default: 6000).

    Retorna
    -------
    str: Query SQL formatada, pronta para execução no Oracle.

    Estratégia
    ----------
    1. Filtra registros do mês informado usando um range de datas, o que permite
       que o otimizador Oracle use índices em START_TIME.
    2. Usa ROW_NUMBER() OVER (PARTITION BY CAMPO1 ORDER BY DBMS_RANDOM.VALUE)
       para selecionar aleatoriamente um único registro por valor de CAMPO1.
    3. Ordena o resultado final por DBMS_RANDOM.VALUE e limita a N registros com
       FETCH FIRST … ROWS ONLY (sintaxe Oracle 12c+).

    Resultado: até N registros, 1 por valor de CAMPO1, distribuídos de forma
    aleatória e estatisticamente representativa do mês.
    """
    _validate_ano_mes(ano_mes)

    campos_in = ", ".join(f"'{c.strip()}'" for c in campos if c.strip())
    if not campos_in:
        raise ValueError("A lista de CAMPO1 não pode estar vazia.")

    return _SAMPLE_SQL_TEMPLATE.format(
        tabela=tabela.strip(),
        ano_mes=ano_mes,
        campos_in=campos_in,
        n_amostras=int(n_amostras),
    )


def _validate_ano_mes(ano_mes: str) -> None:
    """Valida o formato AAAA-MM."""
    try:
        datetime.strptime(ano_mes, "%Y-%m")
    except ValueError:
        raise ValueError(
            f"Formato de mês inválido: '{ano_mes}'. Use AAAA-MM (ex.: 2024-03)."
        )


# ── Conexão Oracle (opcional) ─────────────────────────────────────────────────

def connect_oracle(
    user: str,
    password: str,
    dsn: str,
    thick_mode: bool = False,
) -> "oracledb.Connection":
    """
    Cria e retorna uma conexão com o banco Oracle usando python-oracledb.

    Parâmetros
    ----------
    user:       Usuário Oracle.
    password:   Senha Oracle.
    dsn:        DSN no formato 'host:porta/service_name' ou TNS alias.
    thick_mode: Se True, usa o modo 'Thick' (requer Oracle Client instalado).
                Para a maioria dos ambientes cloud, False (modo Thin) é suficiente.
    """
    try:
        import oracledb
    except ImportError as exc:
        raise ImportError(
            "O pacote 'oracledb' não está instalado. "
            "Execute: pip install oracledb"
        ) from exc

    if thick_mode:
        oracledb.init_oracle_client()

    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    logger.info("Conexão Oracle estabelecida com sucesso.")
    return conn


def run_sample_query(
    conn: "oracledb.Connection",
    ano_mes: str,
    campos: list[str],
    tabela: str = "TABELA",
    n_amostras: int = 6000,
) -> "pd.DataFrame":
    """
    Executa a query de amostragem e retorna um DataFrame pandas.

    Parâmetros
    ----------
    conn:       Conexão Oracle ativa (obtida via `connect_oracle`).
    ano_mes:    Período no formato 'AAAA-MM'.
    campos:     Lista de valores de CAMPO1.
    tabela:     Nome da tabela Oracle.
    n_amostras: Quantidade máxima de registros.

    Retorna
    -------
    pd.DataFrame com colunas: CAMPO1, CAMPO2, START_DATETIME, TRANSCRICAO.
    """
    import pandas as pd

    sql = build_sample_query(
        ano_mes=ano_mes,
        campos=campos,
        tabela=tabela,
        n_amostras=n_amostras,
    )
    logger.info("Executando query de amostragem Oracle…")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    finally:
        cursor.close()

    df = pd.DataFrame(rows, columns=columns)
    logger.info(f"Query retornou {len(df)} registros.")
    return df
