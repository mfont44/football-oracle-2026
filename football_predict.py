# -*- coding: utf-8 -*-
"""
Sistema de predicció de resultats de futbol amb Transfermarkt.
Dataset: csvfiles/ (o csv_files/). Entrena XGBoost i ofereix predictor 1-X-2.
Comentaris en català.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============== CONFIGURACIÓ ==============
# Ruta de la carpeta de CSVs (prova les dues variants)
CSV_DIR = "csvfiles" if os.path.isdir("csvfiles") else "csv_files"
PATH_CLUBS = os.path.join(CSV_DIR, "clubs.csv")
PATH_CLUB_GAMES = os.path.join(CSV_DIR, "club_games.csv")
PATH_GAMES = os.path.join(CSV_DIR, "games.csv")

# Nombre de partits per als rolling averages
ROLLING_WINDOW = 5
# Temporada límit: entrenar fins 2023, provar amb 2024
TRAIN_SEASON_MAX = 2023
TEST_SEASON = 2024

# Variables globals del model i codificació (per al predictor)
model = None
label_encoder = None
feature_cols = None
clubs_df = None
games_full = None


def carregar_dades():
    """
    Carrega els CSVs amb tipus optimitzats per estalviar memòria.
    Retorna: clubs, club_games, games.
    """
    # Tipus explícits per reduir memòria (especialment IDs i enters)
    dtype_clubs = {
        "club_id": "int32",
        "domestic_competition_id": "string",
        "total_market_value": "object",  # pot ser buit o "€50.00m"
        "squad_size": "Int32",
        "average_age": "float32",
        "last_season": "Int32",
    }
    dtype_club_games = {
        "game_id": "int32",
        "club_id": "int32",
        "own_goals": "Int32",
        "opponent_id": "int32",
        "opponent_goals": "Int32",
        "hosting": "string",
        "is_win": "int8",
    }
    dtype_games = {
        "game_id": "int32",
        "competition_id": "string",
        "season": "int32",
        "date": "string",
        "home_club_id": "int32",
        "away_club_id": "int32",
        "home_club_goals": "Int32",
        "away_club_goals": "Int32",
    }

    clubs = pd.read_csv(PATH_CLUBS, dtype=dtype_clubs, low_memory=False)
    club_games = pd.read_csv(PATH_CLUB_GAMES, dtype=dtype_club_games, low_memory=False)
    games = pd.read_csv(
        PATH_GAMES,
        usecols=list(dtype_games.keys()),
        dtype=dtype_games,
        low_memory=False,
    )

    return clubs, club_games, games


def netejar_valor_mercat(serie):
    """
    Converteix columna total_market_value (ex: '€50.00m', '€1.20m') a numèric.
    Valors buits o invàlids es tornen NaN.
    """
    def parse_valor(x):
        if pd.isna(x) or x == "":
            return np.nan
        x = str(x).strip()
        # Eliminar € i espais; detectar m (milions) i k (milers)
        x = re.sub(r"€|\s", "", x)
        if not x or x in ("-", "."):
            return np.nan
        mult = 1.0
        if x.endswith("m") or x.endswith("M"):
            mult = 1e6
            x = x[:-1]
        elif x.endswith("k") or x.endswith("K"):
            mult = 1e3
            x = x[:-1]
        x = x.replace("+", "").replace(",", ".")
        try:
            return float(x) * mult
        except ValueError:
            return np.nan

    return serie.map(parse_valor)


def fusionar_i_rolling(clubs, club_games, games):
    """
    Fusiona club_games amb games (cronologia), calcula punts per partit,
    rolling dels últims 5 partits per club (gols a favor, en contra, punts)
    i uneix total_market_value. Retorna un DataFrame a nivell de partit
    amb característiques local / visitant i target 1/X/2.
    """
    # Merge: cada fila de club_games té game_id; afegim date i season
    cols_games = ["game_id", "season", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]
    meta = games[cols_games].drop_duplicates("game_id")

    cg = club_games.merge(meta, on="game_id", how="inner")

    # Punts: 3 victòria, 1 empat, 0 derrota
    cg["punts"] = np.where(cg["is_win"] == 1, 3, np.where(cg["own_goals"] == cg["opponent_goals"], 1, 0))
    cg["date"] = pd.to_datetime(cg["date"], errors="coerce")
    cg = cg.dropna(subset=["date"]).sort_values(["club_id", "date"]).reset_index(drop=True)

    # Rolling (últims 5 partits) per club; després shift(1) per no incloure el partit actual
    rolling_cols = ["own_goals", "opponent_goals", "punts"]
    for col in rolling_cols:
        cg[f"roll_{col}"] = cg.groupby("club_id")[col].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
        )

    # Valor de mercat: netejar i omplir NaN amb la mitjana
    clubs = clubs.copy()
    clubs["market_value_num"] = netejar_valor_mercat(clubs["total_market_value"])
    mean_mv = clubs["market_value_num"].mean()
    clubs["market_value_num"] = clubs["market_value_num"].fillna(mean_mv)

    # Unir valor de mercat a cada fila de club en el merge
    cg = cg.merge(
        clubs[["club_id", "name", "market_value_num"]].rename(columns={"market_value_num": "club_market_value"}),
        on="club_id",
        how="left",
    )
    # Si faltava algun club, omplir amb la mitjana
    cg["club_market_value"] = cg["club_market_value"].fillna(mean_mv)

    # Dataset a nivell de partit: una fila per game_id amb dades local i visitant
    # Agafem només una fila per (game_id, club_id) per tenir les rolling i market value
    per_club = cg[["game_id", "club_id", "date", "roll_own_goals", "roll_opponent_goals", "roll_punts", "club_market_value"]].drop_duplicates(["game_id", "club_id"])

    # Home: merge games amb per_club per home_club_id
    df = games[["game_id", "season", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]].copy()
    df = df.merge(
        per_club,
        left_on=["game_id", "home_club_id"],
        right_on=["game_id", "club_id"],
        how="inner",
        suffixes=("", "_home"),
    )
    df = df.rename(columns={
        "roll_own_goals": "home_roll_gf",
        "roll_opponent_goals": "home_roll_ga",
        "roll_punts": "home_roll_pts",
        "club_market_value": "home_market_value",
    })
    df = df.drop(columns=["club_id"], errors="ignore")

    # Away: merge amb per_club per away_club_id
    away_feats = per_club.rename(columns={
        "roll_own_goals": "away_roll_gf",
        "roll_opponent_goals": "away_roll_ga",
        "roll_punts": "away_roll_pts",
        "club_market_value": "away_market_value",
    })
    df = df.merge(
        away_feats,
        left_on=["game_id", "away_club_id"],
        right_on=["game_id", "club_id"],
        how="inner",
    )
    df = df.drop(columns=["club_id"], errors="ignore")

    # Target: 1 = local guanya, 0 = empat, 2 = visitant guanya (per al classificador 0/1/2)
    df["resultat"] = np.where(
        df["home_club_goals"] > df["away_club_goals"],
        1,
        np.where(df["home_club_goals"] < df["away_club_goals"], 2, 0),
    )

    # Omplir NaN de rolling amb la mitjana de la columna (robustesa)
    feat_cols = [c for c in df.columns if c.startswith("home_roll_") or c.startswith("away_roll_") or c.startswith("home_market") or c.startswith("away_market")]
    for col in feat_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    return df, feat_cols, cg, clubs, meta


def entrenar_model(df, feature_cols_list):
    """
    Entrenament amb split cronològic: temporades <= 2023 train, 2024 test.
    Retorna model, encoder de labels, accuracy i importància de variables.
    """
    from sklearn.preprocessing import LabelEncoder

    train_df = df[df["season"] <= TRAIN_SEASON_MAX].copy()
    test_df = df[df["season"] == TEST_SEASON].copy()

    # Si no hi ha dades de 2024, fem split temporal (últims 20% per data)
    if test_df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df_sorted = df.dropna(subset=["date"]).sort_values("date")
        n = len(df_sorted)
        train_df = df_sorted.iloc[: int(0.8 * n)]
        test_df = df_sorted.iloc[int(0.8 * n) :]

    X_train = train_df[feature_cols_list].astype(float)
    y_train = train_df["resultat"].astype(int)
    X_test = test_df[feature_cols_list].astype(float)
    y_test = test_df["resultat"].astype(int)

    # Codificar labels 0,1,2 (XGBoost accepta 0,1,2)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test) if len(y_test) > 0 else y_test

    # Entrenar XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train_enc)

    # Mètriques
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred) if len(y_test_enc) > 0 else 0.0
    print("\n--- Resultats del model (split cronològic) ---")
    print(f"Accuracy (test): {acc:.4f}")
    if len(y_test_enc) > 0:
        print(classification_report(y_test_enc, y_pred, target_names=["X (empat)", "1 (local)", "2 (visitant)"]))

    # Importància de variables
    imp = pd.DataFrame({
        "variable": feature_cols_list,
        "importancia": clf.feature_importances_,
    }).sort_values("importancia", ascending=False)
    print("\n--- Importància de variables (top 10) ---")
    print(imp.head(10).to_string(index=False))

    return clf, le, acc, imp


def cercar_equip(nom, clubs_df_ref):
    """
    Cerca un equip per nom. Si no coincideix exactament, fa cerca parcial
    (substring o totes les paraules) i suggereix el més proper.
    Retorna (club_id, nom_canoníc) o (None, suggeriments).
    """
    nom = str(nom).strip()
    if not nom:
        return None, []
    clubs = clubs_df_ref[["club_id", "name"]].drop_duplicates("club_id")
    name_lower = clubs["name"].str.lower()
    nom_lower = nom.lower()
    # Exacte (case-insensitive)
    exact = clubs[name_lower == nom_lower]
    if len(exact) > 0:
        return int(exact.iloc[0]["club_id"]), [exact.iloc[0]["name"]]
    # Parcial: el nom està contingut al nom de l'equip
    partial = clubs[name_lower.str.contains(nom_lower, na=False)]
    if len(partial) == 1:
        return int(partial.iloc[0]["club_id"]), [partial.iloc[0]["name"]]
    if len(partial) > 1:
        # Múltiples coincidències: triar el nom més curt (ex. "Barcelona" -> Futbol Club Barcelona)
        partial = partial.copy()
        partial["_len"] = partial["name"].str.len()
        partial = partial.sort_values("_len").drop(columns=["_len"])
        return int(partial.iloc[0]["club_id"]), partial["name"].tolist()[:5]
    # Cerca per paraules: totes les paraules del query han d’aparèixer al nom (ex: "Real Madrid" -> "Real Madrid Club de Fútbol")
    paraules = [p for p in nom_lower.split() if len(p) > 1]
    if paraules:
        mascara = pd.Series(True, index=clubs.index)
        for p in paraules:
            mascara &= name_lower.str.contains(p, na=False)
        by_words = clubs[mascara]
        if len(by_words) == 1:
            return int(by_words.iloc[0]["club_id"]), [by_words.iloc[0]["name"]]
        if len(by_words) > 1:
            # Múltiples coincidències: triar el nom més curt (sovint el club principal, ex. "Barcelona" -> FC Barcelona)
            by_words = by_words.copy()
            by_words["_len"] = by_words["name"].str.len()
            by_words = by_words.sort_values("_len").drop(columns=["_len"])
            return int(by_words.iloc[0]["club_id"]), by_words["name"].tolist()[:5]
    # Últim recurs: qui comença amb el text
    starts = clubs[name_lower.str.startswith(nom_lower, na=False)]
    if len(starts) > 0:
        return None, starts["name"].tolist()[:5]
    return None, []


def obtenir_ratxa_i_valor(club_id, cg_chrono, clubs_df_ref):
    """
    Retorna la ratxa actual (últims 5 partits del dataset) i valor de mercat
    per a un club_id. cg_chrono ha de tenir columnes club_id, date, own_goals, opponent_goals, punts, club_market_value.
    """
    sub = cg_chrono[cg_chrono["club_id"] == club_id].sort_values("date", ascending=False).head(ROLLING_WINDOW)
    if sub.empty:
        return None, None, None
    # Mitjana dels últims 5 (o menys)
    gf = sub["own_goals"].mean()
    ga = sub["opponent_goals"].mean()
    pts = sub["punts"].mean()
    mv = sub["club_market_value"].iloc[0] if "club_market_value" in sub.columns else clubs_df_ref.loc[clubs_df_ref["club_id"] == club_id, "market_value_num"].iloc[0]
    return gf, ga, pts, mv


def predictor(nom_local, nom_visitant, verbose=True):
    """
    Predicció 1-X-2 a partir dels noms de dos equips.
    Busca IDs, ratxa actual (últims partits del dataset) i valor de mercat,
    i retorna probabilitats en percentatge per 1, X i 2.
    """
    global model, label_encoder, feature_cols, clubs_df, games_full

    if model is None or clubs_df is None or games_full is None:
        if verbose:
            print("Primer cal executar main() per carregar dades i entrenar el model.")
        return None

    # Cerca equips (amb suggeriments si no es troben)
    id_local, sugg_local = cercar_equip(nom_local, clubs_df)
    id_away, sugg_away = cercar_equip(nom_visitant, clubs_df)

    if id_local is None:
        if verbose:
            print(f"Equip no trobat: '{nom_local}'.")
            if sugg_local:
                print("  Suggeriments:", ", ".join(sugg_local))
        return None
    if id_away is None:
        if verbose:
            print(f"Equip no trobat: '{nom_visitant}'.")
            if sugg_away:
                print("  Suggeriments:", ", ".join(sugg_away))
        return None

    # Obtenir ratxa i valor de mercat des del dataset preprocessat (cg amb rolling i market value)
    cg = games_full["cg_chrono"]
    clubs_ref = games_full["clubs"]
    last_stats = games_full["last_stats_per_club"]

    row_home = last_stats[last_stats["club_id"] == id_local]
    row_away = last_stats[last_stats["club_id"] == id_away]
    mean_gf = games_full["mean_roll_gf"]
    mean_ga = games_full["mean_roll_ga"]
    mean_pts = games_full["mean_roll_pts"]
    mean_mv = games_full["mean_market_value"]

    if len(row_home) == 0:
        home_gf, home_ga, home_pts, home_mv = mean_gf, mean_ga, mean_pts, mean_mv
    else:
        r = row_home.iloc[0]
        home_gf = float(r["roll_gf"]) if pd.notna(r["roll_gf"]) else mean_gf
        home_ga = float(r["roll_ga"]) if pd.notna(r["roll_ga"]) else mean_ga
        home_pts = float(r["roll_punts"]) if pd.notna(r["roll_punts"]) else mean_pts
        home_mv = float(r["club_market_value"]) if pd.notna(r["club_market_value"]) else mean_mv

    if len(row_away) == 0:
        away_gf, away_ga, away_pts, away_mv = mean_gf, mean_ga, mean_pts, mean_mv
    else:
        r = row_away.iloc[0]
        away_gf = float(r["roll_gf"]) if pd.notna(r["roll_gf"]) else mean_gf
        away_ga = float(r["roll_ga"]) if pd.notna(r["roll_ga"]) else mean_ga
        away_pts = float(r["roll_punts"]) if pd.notna(r["roll_punts"]) else mean_pts
        away_mv = float(r["club_market_value"]) if pd.notna(r["club_market_value"]) else mean_mv

    # Construir fila de features en el mateix ordre que feature_cols
    X = pd.DataFrame([{
        "home_roll_gf": home_gf,
        "home_roll_ga": home_ga,
        "home_roll_pts": home_pts,
        "home_market_value": home_mv,
        "away_roll_gf": away_gf,
        "away_roll_ga": away_ga,
        "away_roll_pts": away_pts,
        "away_market_value": away_mv,
    }])

    X = X[feature_cols]
    probs = model.predict_proba(X)[0]
    # Ordre de classes en XGBoost: 0, 1, 2 -> en el nostre encoder: 0=X, 1=1 (local), 2=2 (visitant)
    # label_encoder: 0->X, 1->1, 2->2
    p_x = probs[0]
    p_1 = probs[1]
    p_2 = probs[2]
    result = {"1": p_1 * 100, "X": p_x * 100, "2": p_2 * 100}

    if verbose:
        nom_canon_local = clubs_df[clubs_df["club_id"] == id_local]["name"].iloc[0]
        nom_canon_away = clubs_df[clubs_df["club_id"] == id_away]["name"].iloc[0]
        print(f"\nPredicció: {nom_canon_local} vs {nom_canon_away}")
        print(f"  P(1) = {result['1']:.1f}%  |  P(X) = {result['X']:.1f}%  |  P(2) = {result['2']:.1f}%")

    return result


def main():
    """Pipeline complet: carrega, preprocessa, entrena i deixa preparat el predictor."""
    global model, label_encoder, feature_cols, clubs_df, games_full

    print("Carregant dades...")
    clubs, club_games, games = carregar_dades()
    clubs_df = clubs

    print("Preprocessant (merge, rolling, valor de mercat)...")
    df, feature_cols_list, cg_chrono, clubs_processed, meta = fusionar_i_rolling(clubs, club_games, games)

    # Omplir NaN restants amb mitjana
    for col in feature_cols_list:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Estadístiques per al predictor (últim estat per club)
    cg_sorted = cg_chrono.sort_values(["club_id", "date"])
    cg_sorted["roll_gf"] = cg_sorted.groupby("club_id")["own_goals"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    cg_sorted["roll_ga"] = cg_sorted.groupby("club_id")["opponent_goals"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    cg_sorted["roll_punts"] = cg_sorted.groupby("club_id")["punts"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    last_stats = cg_sorted.groupby("club_id").last().reset_index()[["club_id", "roll_gf", "roll_ga", "roll_punts", "club_market_value"]]
    last_stats.columns = ["club_id", "roll_gf", "roll_ga", "roll_punts", "club_market_value"]

    mean_roll_gf = last_stats["roll_gf"].mean()
    mean_roll_ga = last_stats["roll_ga"].mean()
    mean_roll_pts = last_stats["roll_punts"].mean()
    mean_mv = last_stats["club_market_value"].mean()

    games_full = {
        "cg_chrono": cg_chrono,
        "clubs": clubs_processed,
        "last_stats_per_club": last_stats,
        "mean_roll_gf": mean_roll_gf,
        "mean_roll_ga": mean_roll_ga,
        "mean_roll_pts": mean_roll_pts,
        "mean_market_value": mean_mv,
    }

    print("Entrenant model (XGBoost, split cronològic)...")
    model, label_encoder, acc, importance = entrenar_model(df, feature_cols_list)
    feature_cols = feature_cols_list

    # Exemple de predicció (noms que coincideixen amb el dataset: Real Madrid, Barcelona, etc.)
    print("\n--- Exemple: predictor('Real Madrid', 'Barcelona') ---")
    predictor("Real Madrid", "Barcelona")

    return model, df, feature_cols_list


if __name__ == "__main__":
    main()
