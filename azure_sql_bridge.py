# -*- coding: utf-8 -*-
"""
Puja els CSVs resumits a Azure SQL Database.
Connexió amb SQLAlchemy + pyodbc. Driver: nom exacte del sistema (amb claus).
Credencials: variables d'entorn AZURE_SQL_USER i AZURE_SQL_PASSWORD.
"""

import os
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine


def pujar_dades():
    # Dades confirmades per l'usuari
    host = "predict1.database.windows.net"
    db = "football-oracle-db"
    # Nom exacte del driver ODBC (amb claus, tal com apareix al sistema)
    driver = "{ODBC Driver 18 for SQL Server}"

    # Credencials des de variables d'entorn (no posar contrasenyes al codi)
    user = os.environ.get("AZURE_SQL_USER")
    password = os.environ.get("AZURE_SQL_PASSWORD")
    if not user or not password:
        print("Configura AZURE_SQL_USER i AZURE_SQL_PASSWORD (variables d'entorn).")
        return

    print(f"Connexió a {host} amb driver {driver}...")

    params = urllib.parse.quote_plus(
        f"DRIVER={driver};SERVER={host};DATABASE={db};UID={user};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=yes;LoginTimeout=60;"
    )

    try:
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        fitxers = {
            "multi_league_2026_summary": "csvfiles/multi_league_2026_summary.csv",
            "espn_2025_summary": "csvfiles/espn_2025_summary.csv",
            "statsbomb_summary": "csvfiles/statsbomb_summary.csv",
            "clubs": "csvfiles/clubs.csv"
        }

        for taula, path in fitxers.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.to_sql(taula, engine, if_exists="replace", index=False, method="multi", chunksize=100)
                print(f"✅ {taula} OK")
            else:
                print(f"❌ No trobat: {path}")
    except Exception as e:
        print(f"❌ ERROR CRÍTIC: {e}")


if __name__ == "__main__":
    pujar_dades()
