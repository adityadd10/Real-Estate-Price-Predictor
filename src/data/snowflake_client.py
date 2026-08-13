"""Thin wrapper around the Snowflake Python connector.

Two operations only: read a table into a pandas DataFrame, and write a
DataFrame out as a table. Everything else (schema/warehouse bootstrap, the
actual table layout) lives in load_to_snowflake.py so this module stays a
reusable, boring client.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from src.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Iterator[snowflake.connector.SnowflakeConnection]:
    if not settings.snowflake_configured:
        raise RuntimeError(
            "Snowflake is not configured. Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER and "
            "SNOWFLAKE_PASSWORD (see .env.example)."
        )
    conn = snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        password=settings.snowflake_password,
        role=settings.snowflake_role,
        warehouse=settings.snowflake_warehouse,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,
    )
    try:
        yield conn
    finally:
        conn.close()


def read_table(table_name: str) -> pd.DataFrame:
    """Read an entire table into a DataFrame. Fine for this project's data sizes
    (a few thousand rows) — for anything larger you'd push filters into SQL instead.

    Column names round-trip exactly as written (see write_table) — no case
    normalization here, since this project's feature names are case-sensitive
    (e.g. "bedRoom", "agePossession") and write_pandas quotes identifiers to
    preserve exactly that casing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM "{table_name.upper()}"')
            return cur.fetch_pandas_all()


def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run an arbitrary SELECT and return the result as a DataFrame. Used for
    queries that should be filtered warehouse-side (e.g. nearby-property search)
    rather than pulled in full and filtered in pandas."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetch_pandas_all()


def write_table(df: pd.DataFrame, table_name: str, overwrite: bool = True) -> None:
    """Write a DataFrame to Snowflake, creating the table if needed. Column names
    are written exactly as given (write_pandas quotes identifiers), so casing
    like "bedRoom" or spaces like "servant room" survive the round trip."""
    with get_connection() as conn:
        write_pandas(
            conn,
            df,
            table_name.upper(),
            auto_create_table=True,
            overwrite=overwrite,
        )
    logger.info("Wrote %d rows to %s", len(df), table_name)


def execute(sql: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
