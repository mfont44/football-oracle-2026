# -*- coding: utf-8 -*-
"""
Script de suport: elimina variables amb feature_importance < 0.012 i les altament correlacionades.
Llegeix csvfiles/feature_importance.csv i opcionalment csvfiles/features_correlated_drop.txt.
Escriu csvfiles/features_pruned.txt. En la propera execució, football_pro_model main() usa només aquestes features.
"""

from __future__ import annotations

import os

import pandas as pd

CSV_DIR = "csvfiles" if os.path.isdir("csvfiles") else "csv_files"
PATH_FEATURE_IMPORTANCE = os.path.join(CSV_DIR, "feature_importance.csv")
PATH_PRUNED_FEATURES = os.path.join(CSV_DIR, "features_pruned.txt")
PATH_FEATURES_CORRELATED_DROP = os.path.join(CSV_DIR, "features_correlated_drop.txt")
THRESHOLD = 0.012
TOP_N_KEEP_ODDS = 10  # Quotes individuals es mantenen només si estan al Top N
ODDS_INDIVIDUAL_PATTERNS = ("AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "WHH", "WHD", "WHA")


def _is_odds_individual(var: str) -> bool:
    return any(p in var for p in ODDS_INDIVIDUAL_PATTERNS)


def main() -> None:
    if not os.path.isfile(PATH_FEATURE_IMPORTANCE):
        print(f"No s'ha trobat {PATH_FEATURE_IMPORTANCE}. Executa abans football_pro_model.py per generar-lo.")
        return

    df = pd.read_csv(PATH_FEATURE_IMPORTANCE, low_memory=False)
    if "variable" not in df.columns or "importancia" not in df.columns:
        print("El CSV ha de tenir columnes 'variable' i 'importancia'.")
        return

    df = df.sort_values("importancia", ascending=False).reset_index(drop=True)
    top10_vars = set(df.head(TOP_N_KEEP_ODDS)["variable"].tolist())

    # Variables marcades per correlació (football_pro_model les escriu a features_correlated_drop.txt)
    correlated_drop: set[str] = set()
    if os.path.isfile(PATH_FEATURES_CORRELATED_DROP):
        with open(PATH_FEATURES_CORRELATED_DROP, "r", encoding="utf-8") as f:
            correlated_drop = {line.strip() for line in f if line.strip()}

    # Conservar: importància >= THRESHOLD; no correlacionades; i si és quota individual, només si està al Top 10
    kept = []
    dropped_low = []
    dropped_odds = []
    dropped_corr = []
    for _, row in df.iterrows():
        var, imp = row["variable"], row["importancia"]
        if imp < THRESHOLD:
            dropped_low.append(var)
            continue
        if var in correlated_drop:
            dropped_corr.append(var)
            continue
        if _is_odds_individual(var) and var not in top10_vars:
            dropped_odds.append(var)
            continue
        kept.append(var)
    dropped = dropped_low + dropped_odds + dropped_corr

    os.makedirs(CSV_DIR, exist_ok=True)
    with open(PATH_PRUNED_FEATURES, "w", encoding="utf-8") as f:
        for v in kept:
            f.write(v + "\n")

    print(f"Umbral d'importància: {THRESHOLD}")
    print(f"Variables conservades: {len(kept)}")
    print(f"Eliminades (importància < {THRESHOLD}): {len(dropped_low)}")
    print(f"Eliminades (correlacionades): {len(dropped_corr)}")
    print(f"Eliminades (quotes individuals fora del Top {TOP_N_KEEP_ODDS}): {len(dropped_odds)}")
    if dropped:
        print("Eliminades:", ", ".join(dropped[:15]) + ("..." if len(dropped) > 15 else ""))
    print(f"Llista guardada a {PATH_PRUNED_FEATURES}")


if __name__ == "__main__":
    main()
