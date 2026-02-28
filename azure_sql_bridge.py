# -*- coding: utf-8 -*-
"""
Puja només els fitxers essencials per al model a Azure SQL Database.

FITXERS_OK (carpeta csvfiles/ o csv_files/):
- clubs.csv
- club_games.csv
- games.csv
- players.csv
- espn_2025_summary.csv
- multi_league_2026_summary.csv
- statsbomb_summary.csv
- competitions.csv

Connexió amb SQLAlchemy + pyodbc.
Credencials: variables d'entorn AZURE_SQL_USER i AZURE_SQL_PASSWORD.
"""

import os
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine

# Nom de taula per cada fitxer essencial que es vol pujar
FITXERS_OK = {
    "clubs.csv": "clubs",
    "club_games.csv": "club_games",
    "games.csv": "games",
    "players.csv": "players",
    "espn_2025_summary.csv": "espn_2025_summary",
    "multi_league_2026_summary.csv": "multi_league_2026_summary",
    "statsbomb_summary.csv": "statsbomb_summary",
    "competitions.csv": "competitions",
}

# Pujada estable a Azure SQL Serverless. SQL Server límit 2100 paràmetres per INSERT.
MAX_PARAMS_PER_INSERT = 2100
CHUNKSIZE_SQL = 100


def _chunksize_sql(num_columns: int) -> int:
    """Chunksize segur per to_sql: files × columnes <= 2100."""
    if num_columns <= 0:
        return 1
    return min(CHUNKSIZE_SQL, max(1, MAX_PARAMS_PER_INSERT // num_columns))


def _netejar_columnes_duplicades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el CSV té capçaleres repetides, deixa només la primera ocurència de cada nom.
    Evita l'error 07002 (COUNT field incorrect) en fer to_sql.
    """
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()]
    return df


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

    # Directori base de CSVs
    if os.path.isdir("csvfiles"):
        base_dir = "csvfiles"
    elif os.path.isdir("csv_files"):
        base_dir = "csv_files"
    else:
        print("No s'ha trobat cap directori de CSVs ('csvfiles' ni 'csv_files').")
        return

    print(f"Carregant CSVs essencials des de '{base_dir}'...")
    print(f"Connexió a {host} amb driver {driver}...")

    params = urllib.parse.quote_plus(
        f"DRIVER={driver};SERVER={host};DATABASE={db};UID={user};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=yes;LoginTimeout=60;"
    )

    try:
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        total = 0
        for fname, table in FITXERS_OK.items():
            full_path = os.path.join(base_dir, fname)
            if not os.path.exists(full_path):
                print(f"❌ No trobat (s'ignora): {full_path}")
                continue

            try:
                print(f"Pujant '{full_path}' -> taula '{table}' ...")
                df = pd.read_csv(full_path, low_memory=False)
                # Netejar columnes duplicades
                df = _netejar_columnes_duplicades(df)
                # Normalitzar noms de columna: espais i punts -> "_"
                df.columns = [str(c).replace(" ", "_").replace(".", "_") for c in df.columns]

                sql_chunk = _chunksize_sql(len(df.columns))
                df.to_sql(
                    table,
                    engine,
                    if_exists="replace",  # sempre neteja la taula
                    index=False,
                    method="multi",
                    chunksize=sql_chunk,
                )
                print(f"✅ {table} OK ({len(df)} files)")
                total += 1
            except Exception as e:
                print(f"❌ ERROR a '{full_path}' -> '{table}': {e}")

        if total == 0:
            print("No s'ha pogut pujar cap fitxer essencial.")
        else:
            print(f"Pujada completada. {total} taules essencials actualitzades.")

    except Exception as e:
        print(f"❌ ERROR CRÍTIC: {e}")


if __name__ == "__main__":
    pujar_dades()
