# -*- coding: utf-8 -*-
"""
Exporta els CSVs resumits a PostgreSQL (Azure o altre).
Llegeix credencials de st.secrets (Streamlit Cloud) o variables d'entorn (local).
No desis mai contrasenyes al codi.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# Fitxers resum i noms de taula
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csvfiles")
SUMMARY_FILES = [
    ("espn_2025_summary.csv", "espn_2025_summary"),
    ("statsbomb_summary.csv", "statsbomb_summary"),
    ("multi_league_2026_summary.csv", "multi_league_2026_summary"),
]


def _get_connection_string() -> Optional[str]:
    """
    Obté la URL de connexió PostgreSQL.
    Ordre: DATABASE_URL (env) → POSTGRES_* (env) → st.secrets (Streamlit Cloud).
    """
    # 1) Variable d'entorn URL completa (local o Cloud)
    url = os.environ.get("DATABASE_URL")
    if url:
        if not url.startswith("postgresql://") and not url.startswith("postgres://"):
            url = "postgresql://" + url.lstrip("/")
        return url

    # 2) Variables d'entorn individuals (local)
    host = os.environ.get("POSTGRES_HOST")
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")
    port = os.environ.get("POSTGRES_PORT", "5432")
    if host and user and password and db:
        return f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode=require"

    # 3) Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        secrets = getattr(st, "secrets", None)
        if secrets:
            db_secret = secrets.get("database") or secrets.get("postgres") or {}
            url = db_secret.get("url") or db_secret.get("connection_string")
            if url:
                return url
            host = db_secret.get("host") or secrets.get("postgres_host")
            user = db_secret.get("user") or secrets.get("postgres_user")
            password = db_secret.get("password") or secrets.get("postgres_password")
            db = db_secret.get("database") or db_secret.get("db") or secrets.get("postgres_db")
            port = str(db_secret.get("port") or secrets.get("postgres_port", "5432"))
            if host and user and password and db:
                return f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode=require"
    except Exception:
        pass

    return None


def get_engine() -> Optional[Engine]:
    """Crea l'engine de SQLAlchemy si hi ha credencials."""
    url = _get_connection_string()
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True)


def sync_summaries_to_cloud(engine: Optional[Engine] = None, if_exists: str = "replace") -> dict[str, bool]:
    """
    Puja els tres CSVs resumits a PostgreSQL amb to_sql().
    if_exists: 'replace' (per defecte), 'append' o 'fail'.
    Retorna { "nom_taula": True/False } segons èxit.
    """
    if engine is None:
        engine = get_engine()
    if engine is None:
        print("No s'han trobat credencials (DATABASE_URL, POSTGRES_* o st.secrets). No es fa res.")
        return {}

    results: dict[str, bool] = {}
    for filename, table_name in SUMMARY_FILES:
        path = os.path.join(CSV_DIR, filename)
        if not os.path.isfile(path):
            print(f"  Fitxer no trobat: {path}")
            results[table_name] = False
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            df.to_sql(table_name, engine, method="multi", index=False, if_exists=if_exists, chunksize=500)
            print(f"  {table_name}: {len(df)} files exportades.")
            results[table_name] = True
        except Exception as e:
            print(f"  Error en {table_name}: {e}")
            results[table_name] = False

    return results


def main() -> None:
    print("Sincronitzant resums a PostgreSQL (Azure)...")
    engine = get_engine()
    if engine is None:
        print("Configura DATABASE_URL o POSTGRES_HOST/USER/PASSWORD/DB (o st.secrets en Streamlit).")
        return
    sync_summaries_to_cloud(engine)
    print("Fet.")


if __name__ == "__main__":
    main()
