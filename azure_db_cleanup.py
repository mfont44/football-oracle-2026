# -*- coding: utf-8 -*-
"""
Esborra TOTES les taules d'usuari de la base de dades Azure SQL.
Connexió: mateixa lògica que azure_sql_bridge.py (urllib.parse.quote_plus + create_engine).
Credencials: variables d'entorn AZURE_SQL_USER i AZURE_SQL_PASSWORD.

Seguretat: demana confirmació abans d'executar el borrat.
"""

import os
import urllib.parse

from sqlalchemy import create_engine, text


def esborrar_totes_taules() -> None:
    host = "predict1.database.windows.net"
    db = "football-oracle-db"
    driver = "{ODBC Driver 18 for SQL Server}"

    user = os.environ.get("AZURE_SQL_USER")
    password = os.environ.get("AZURE_SQL_PASSWORD")
    if not user or not password:
        print("Configura AZURE_SQL_USER i AZURE_SQL_PASSWORD (variables d'entorn).")
        return

    params = urllib.parse.quote_plus(
        f"DRIVER={driver};SERVER={host};DATABASE={db};UID={user};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=yes;LoginTimeout=60;"
    )

    print(f"Connexió a {host} / {db}...")
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sys.tables WHERE type = 'U'"))
        taules = [row[0] for row in result]

    if not taules:
        print("No hi ha cap taula d'usuari a la base de dades.")
        return

    print(f"\nS'han trobat {len(taules)} taules:")
    for t in taules:
        print(f"  - {t}")

    confirmacio = input("\nPer esborrar TOTES aquestes taules, escriu exactament: SI\n> ")
    if confirmacio.strip() != "SI":
        print("No s'ha escrit 'SI'. Operació cancel·lada.")
        return

    print("\nEsborrant taules...")
    with engine.connect() as conn:
        for nom in taules:
            try:
                conn.execute(text(f"DROP TABLE [{nom}]"))
                conn.commit()
                print(f"  ✅ Esborrada: {nom}")
            except Exception as e:
                print(f"  ❌ Error en '{nom}': {e}")
                conn.rollback()

    print("\nFi.")


if __name__ == "__main__":
    esborrar_totes_taules()
