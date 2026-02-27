# -*- coding: utf-8 -*-
"""
Sistema de predicció Over 2.5 gols (model millorat).
- Squad Value: valor de plantilla des de players.csv (market_value_in_eur per current_club_id).
- EWM (span=5): form amb mitjana exponencial, més pes als partits recents.
- new_manager_effect: 1 si l’entrenador ha canviat respecte al partit anterior (own_manager_name a club_games).
- mv_ratio: home_market_value / away_market_value (valor relatiu, no només absolut).
- Comentaris en català. Type hints i gestió d’errors.

Impacte en Feature Importance (resum típic):
  mv_ratio domina (~0.36) perquè captura la desigualtat entre equips; home/away_market_value
  queden per sota (~0.17/0.09). Les rolling (EWM) tenen importància similar entre elles (~0.05).
  home_new_manager_effect aporta (~0.04); el model aprèn part dels efectes del canvi d’entrenador.
"""

from __future__ import annotations

import os
import re
import urllib.parse
import warnings
from datetime import date
from typing import Any, Optional
import difflib

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============== CONFIGURACIÓ ==============
CSV_DIR = "csvfiles" if os.path.isdir("csvfiles") else "csv_files"
PATH_CLUBS = os.path.join(CSV_DIR, "clubs.csv")
PATH_CLUB_GAMES = os.path.join(CSV_DIR, "club_games.csv")
PATH_GAMES = os.path.join(CSV_DIR, "games.csv")
PATH_PLAYERS = os.path.join(CSV_DIR, "players.csv")
PATH_COMPETITIONS = os.path.join(CSV_DIR, "competitions.csv")
PATH_ESPN_SUMMARY = os.path.join(CSV_DIR, "espn_2025_summary.csv")
# Resum multi-lliga (LaLiga, Premier, Bundes, League1, SerieA): avg_corners, avg_cards, market_expectation per club_id
PATH_MULTI_LEAGUE_2026 = os.path.join(CSV_DIR, "multi_league_2026_summary.csv")
PATH_STATSBOMB_SUMMARY = os.path.join(CSV_DIR, "statsbomb_summary.csv")
PATH_FEATURE_IMPORTANCE = os.path.join(CSV_DIR, "feature_importance.csv")
PATH_PRUNED_FEATURES = os.path.join(CSV_DIR, "features_pruned.txt")
PATH_FEATURES_CORRELATED_DROP = os.path.join(CSV_DIR, "features_correlated_drop.txt")

# ---------- Azure SQL (càrrega híbrida amb st.secrets) ----------
def _get_azure_engine() -> Optional[Any]:
    """Engine SQLAlchemy per Azure SQL si st.secrets[\"azure_sql\"] està disponible (Streamlit Cloud)."""
    try:
        import streamlit as st
        from sqlalchemy import create_engine
        s = getattr(st, "secrets", None)
        if not s or not s.get("azure_sql"):
            return None
        az = s["azure_sql"]
        host = az.get("host") or "predict1.database.windows.net"
        db = az.get("database") or "football-oracle-db"
        driver = az.get("driver") or "{ODBC Driver 18 for SQL Server}"
        user = az.get("user")
        password = az.get("password")
        if not user or not password:
            return None
        params = urllib.parse.quote_plus(
            f"DRIVER={driver};SERVER={host};DATABASE={db};UID={user};PWD={password};"
            "Encrypt=yes;TrustServerCertificate=yes;LoginTimeout=60;"
        )
        return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", pool_pre_ping=True)
    except Exception:
        return None


def get_data_from_azure(table_name: str) -> Optional[pd.DataFrame]:
    """Retorna DataFrame amb SELECT * FROM table_name des d'Azure SQL, o None si falla."""
    engine = _get_azure_engine()
    if engine is None:
        return None
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", engine)
    except Exception:
        return None


def _load_summary_df(table_name: str, csv_path: str) -> Optional[pd.DataFrame]:
    """
    Càrrega híbrida per als resums:
    - Si hi ha st.secrets["azure_sql"], només es consulta Azure (sense intentar CSV local).
    - Si NO hi ha secrets, es fa servir el CSV local si existeix.
    """
    try:
        import streamlit as st  # només per comprovar si hi ha secrets

        s = getattr(st, "secrets", None)
        has_azure = bool(s and s.get("azure_sql"))
    except Exception:
        has_azure = False

    if has_azure:
        df = get_data_from_azure(table_name)
        if df is not None and not df.empty:
            return df
        # En mode Azure, no fem fallback a fitxers locals
        return None

    if os.path.isfile(csv_path):
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except Exception:
            return None
    return None

# Quotes individuals (excloses de feat_cols; es manté només market_expectation / ml_market_expectation)
ODDS_INDIVIDUAL_BLACKLIST = frozenset(
    {"AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "WHH", "WHD", "WHA"}
)

ROLLING_WINDOW = 5  # usat a main() per last_stats (alias short)
SHORT_ROLL = 4   # explosivitat (moment actual)
LONG_ROLL = 10   # consistència (base)
MIN_SEASON = 2018   # filtre dràstic: cap partit anterior (model "fresc")
TRAIN_SEASON_MAX = 2023
TEST_SEASON = 2024

# Estat global del model (per al predictor i per l'app)
model: Any = None
feature_cols: Optional[list[str]] = None
clubs_df: Optional[pd.DataFrame] = None
games_full: Optional[dict[str, Any]] = None
# Mètriques del darrer test (per a la UI)
test_accuracy: float = 0.0
test_confusion_matrix: Optional[np.ndarray] = None
feature_importance_df: Optional[pd.DataFrame] = None
baseline_accuracy: Optional[float] = None


def carregar_dades() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega els CSVs (clubs, club_games, games, players) amb tipus optimitzats.
    Millora: inclou players.csv per poder calcular el Squad Value després.
    """
    dtype_clubs = {
        "club_id": "int32",
        "domestic_competition_id": "string",
        "total_market_value": "object",
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
        "own_manager_name": "string",  # per detecció canvi d'entrenador
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
    dtype_players = {
        "player_id": "int32",
        "current_club_id": "Int32",
        "market_value_in_eur": "float64",
        "name": "string",
    }

    # Si hi ha st.secrets["azure_sql"], totes les dades han de venir d'Azure (sense CSV locals)
    try:
        import streamlit as st  # només per comprovar secrets

        s = getattr(st, "secrets", None)
        has_azure = bool(s and s.get("azure_sql"))
    except Exception:
        has_azure = False

    if has_azure:
        clubs = get_data_from_azure("clubs")
        club_games = get_data_from_azure("club_games")
        games = get_data_from_azure("games")
        players = get_data_from_azure("players")
        if clubs is None or club_games is None or games is None or players is None:
            raise FileNotFoundError(
                "No s'han trobat una o més taules a Azure SQL "
                "('clubs', 'club_games', 'games', 'players')."
            )
        return clubs, club_games, games, players

    # Mode local: lectura des de CSV
    clubs = pd.read_csv(PATH_CLUBS, dtype=dtype_clubs, low_memory=False)
    club_games = pd.read_csv(PATH_CLUB_GAMES, dtype=dtype_club_games, low_memory=False)
    games = pd.read_csv(
        PATH_GAMES,
        usecols=list(dtype_games.keys()),
        dtype=dtype_games,
        low_memory=False,
    )
    # Millora: carregar jugadors per calcular valor de plantilla (Squad Value)
    if not os.path.isfile(PATH_PLAYERS):
        raise FileNotFoundError(f"No s'ha trobat {PATH_PLAYERS}. Necessari per al Squad Value.")
    players = pd.read_csv(
        PATH_PLAYERS,
        usecols=["player_id", "current_club_id", "market_value_in_eur", "name"],
        dtype=dtype_players,
        low_memory=False,
    )

    return clubs, club_games, games, players


def computar_squad_value(players: pd.DataFrame) -> pd.Series:
    """
    Calcula el valor de la plantilla (Squad Value) sumant market_value_in_eur
    per current_club_id. Si un jugador té market_value_in_eur NaN, es tracta com 0.
    Retorna una Series amb index = current_club_id (club_id), valor = suma en EUR.
    """
    df = players[["current_club_id", "market_value_in_eur"]].copy()
    df["current_club_id"] = pd.to_numeric(df["current_club_id"], errors="coerce")
    df = df.dropna(subset=["current_club_id"])
    df["market_value_in_eur"] = pd.to_numeric(df["market_value_in_eur"], errors="coerce").fillna(0.0)
    squad_value = df.groupby("current_club_id", as_index=True)["market_value_in_eur"].sum()
    return squad_value.astype(np.float64)


def netejar_valor_mercat(serie: pd.Series) -> pd.Series:
    """Converteix total_market_value (ex: '€50.00m') a numèric. Fallback si no hi ha players."""
    def parse_valor(x: Any) -> float:
        if pd.isna(x) or x == "":
            return np.nan
        x = str(x).strip()
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


def fusionar_i_rolling(
    clubs: pd.DataFrame,
    club_games: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fusiona club_games amb games, calcula rolling (últims 5 partits) i uneix el valor de mercat.
    Millora: utilitza Squad Value (suma de market_value_in_eur per club des de players.csv)
    en lloc de total_market_value de clubs.csv. Equips sense jugadors al dataset: es fa servir
    la mitjana global (gestió d'errors) per no perdre partits.

    També integra:
    - competition_type (Lliga / Copa / Champions) des de competitions.csv
    - micro-estadístiques ESPN 2024-2025 (possessió, tirs, índex ofensiu/defensiu) si espn_2025_summary.csv existeix.
    """
    cols_games = [
        "game_id",
        "season",
        "date",
        "competition_id",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
    ]
    meta = games[cols_games].drop_duplicates("game_id")
    cg = club_games.merge(meta, on="game_id", how="inner")

    cg["punts"] = np.where(cg["is_win"] == 1, 3, np.where(cg["own_goals"] == cg["opponent_goals"], 1, 0))
    cg["date"] = pd.to_datetime(cg["date"], errors="coerce")
    cg = cg.dropna(subset=["date"]).sort_values(["club_id", "date"]).reset_index(drop=True)

    # new_manager_effect, days_rest i club_market_value (abans del split Home/Away)
    if "own_manager_name" in cg.columns:
        cg["prev_manager"] = cg.groupby("club_id")["own_manager_name"].shift(1)
        cg["new_manager_effect"] = (
            (cg["own_manager_name"] != cg["prev_manager"]) & cg["own_manager_name"].notna() & cg["prev_manager"].notna()
        ).astype(np.int8)
        cg["new_manager_effect"] = cg["new_manager_effect"].fillna(0).astype(np.int8)
    else:
        cg["new_manager_effect"] = 0
    cg["prev_match_date"] = cg.groupby("club_id")["date"].shift(1)
    cg["days_rest"] = (cg["date"] - cg["prev_match_date"]).dt.days
    median_rest = float(cg["days_rest"].median()) if cg["days_rest"].notna().any() else 7.0
    cg["days_rest"] = cg["days_rest"].fillna(median_rest).clip(lower=0).astype(np.int32)
    squad_value = computar_squad_value(players)
    mean_squad = float(squad_value.mean())
    clubs_copy = clubs.copy()
    clubs_copy["market_value_num"] = netejar_valor_mercat(clubs_copy["total_market_value"])
    fallback_series = clubs_copy.set_index("club_id")["market_value_num"]
    all_club_ids = clubs["club_id"].unique()
    club_market_map = squad_value.reindex(all_club_ids).fillna(fallback_series).fillna(mean_squad)
    cg["club_market_value"] = cg["club_id"].map(club_market_map).fillna(mean_squad)
    clubs_processed = clubs.copy()
    clubs_processed["market_value_num"] = clubs_processed["club_id"].map(club_market_map).fillna(mean_squad)

    # Millora 1: Ratxes específiques per camp (Home/Away) — Doble EWM short (4) i long (10).
    # roll_gf_home = gols a favor quan l'equip juga a casa; roll_gf_away = a fora.
    if "hosting" in cg.columns:
        host_str = cg["hosting"].astype(str).str.strip().str.lower()
        is_home = host_str.isin(["home", "1", "true", "yes", "h"])
        cg_home = cg.loc[is_home].sort_values(["club_id", "date"]).copy()
        cg_away = cg.loc[~is_home].sort_values(["club_id", "date"]).copy()
    else:
        # Fallback: repartir 50/50 per game_id (primer club = home, segon = away)
        first_club = cg.drop_duplicates("game_id", keep="first").set_index("game_id")["club_id"]
        cg = cg.copy()
        cg["_is_home"] = cg.apply(lambda r: r["club_id"] == first_club.get(r["game_id"], -1), axis=1)
        cg_home = cg.loc[cg["_is_home"]].sort_values(["club_id", "date"]).copy()
        cg_away = cg.loc[~cg["_is_home"]].sort_values(["club_id", "date"]).copy()
        cg = cg.drop(columns=["_is_home"], errors="ignore")

    for col, suffix in [("own_goals", "gf"), ("opponent_goals", "ga"), ("punts", "pts")]:
        # Home
        cg_home[f"roll_{col}_home_short"] = cg_home.groupby("club_id")[col].transform(
            lambda x: x.shift(1).ewm(span=SHORT_ROLL, adjust=False).mean()
        )
        cg_home[f"roll_{col}_home_long"] = cg_home.groupby("club_id")[col].transform(
            lambda x: x.shift(1).ewm(span=LONG_ROLL, adjust=False).mean()
        )
        # Away
        cg_away[f"roll_{col}_away_short"] = cg_away.groupby("club_id")[col].transform(
            lambda x: x.shift(1).ewm(span=SHORT_ROLL, adjust=False).mean()
        )
        cg_away[f"roll_{col}_away_long"] = cg_away.groupby("club_id")[col].transform(
            lambda x: x.shift(1).ewm(span=LONG_ROLL, adjust=False).mean()
        )

    # merge_asof: afegir al home la forma a fora (més recent) per calcular home_advantage
    # Fem merge_asof per club_id per evitar problemes d'ordenació amb pandas
    roll_away_cols = [c for c in cg_away.columns if c.startswith("roll_") and "_away_" in c]
    cg_home["date"] = pd.to_datetime(cg_home["date"], utc=False)
    cg_away_dt = cg_away.copy()
    cg_away_dt["date"] = pd.to_datetime(cg_away_dt["date"], utc=False)
    away_lookup = cg_away_dt[["club_id", "date"] + roll_away_cols].drop_duplicates(["club_id", "date"])
    home_parts = []
    for cid, g in cg_home.groupby("club_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        right = away_lookup[away_lookup["club_id"] == cid].sort_values("date").reset_index(drop=True)
        if right.empty:
            home_parts.append(g)
            continue
        merged = pd.merge_asof(g, right, on="date", direction="backward")
        home_parts.append(merged)
    cg_home = pd.concat(home_parts, ignore_index=True)
    cg_home["home_advantage_pts_short"] = (cg_home["roll_punts_home_short"] - cg_home.get("roll_punts_away_short", cg_home["roll_punts_home_short"])).fillna(0)
    cg_home["home_advantage_pts_long"] = (cg_home["roll_punts_home_long"] - cg_home.get("roll_punts_away_long", cg_home["roll_punts_home_long"])).fillna(0)
    cg_home["home_advantage_gf_short"] = (cg_home["roll_own_goals_home_short"] - cg_home.get("roll_own_goals_away_short", cg_home["roll_own_goals_home_short"])).fillna(0)
    cg_home["home_advantage_ga_short"] = (cg_home["roll_opponent_goals_home_short"] - cg_home.get("roll_opponent_goals_away_short", cg_home["roll_opponent_goals_home_short"])).fillna(0)

    roll_home_cols = [c for c in cg_home.columns if c.startswith("roll_") and "_home_" in c]
    home_lookup = cg_home[["club_id", "date"] + roll_home_cols].drop_duplicates(["club_id", "date"])
    cg_away["date"] = pd.to_datetime(cg_away["date"], utc=False)
    away_parts = []
    for cid, g in cg_away.groupby("club_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        right = home_lookup[home_lookup["club_id"] == cid].sort_values("date").reset_index(drop=True)
        if right.empty:
            away_parts.append(g)
            continue
        merged = pd.merge_asof(g, right, on="date", direction="backward")
        away_parts.append(merged)
    cg_away = pd.concat(away_parts, ignore_index=True)
    cg_away["away_advantage_pts_short"] = (cg_away["roll_punts_away_short"] - cg_away.get("roll_punts_home_short", cg_away["roll_punts_away_short"])).fillna(0)
    cg_away["away_advantage_pts_long"] = (cg_away["roll_punts_away_long"] - cg_away.get("roll_punts_home_long", cg_away["roll_punts_away_long"])).fillna(0)
    cg_away["away_advantage_gf_short"] = (cg_away["roll_own_goals_away_short"] - cg_away.get("roll_own_goals_home_short", cg_away["roll_own_goals_away_short"])).fillna(0)
    cg_away["away_advantage_ga_short"] = (cg_away["roll_opponent_goals_away_short"] - cg_away.get("roll_opponent_goals_home_short", cg_away["roll_opponent_goals_away_short"])).fillna(0)

    # Portar roll i advantage de nou a cg: merge per (game_id, club_id)
    cg = cg.merge(
        cg_home[["game_id", "club_id"] + roll_home_cols + [c for c in cg_home.columns if "advantage" in c and c.startswith("home_")]],
        on=["game_id", "club_id"],
        how="left",
    )
    cg = cg.merge(
        cg_away[["game_id", "club_id"] + roll_away_cols + [c for c in cg_away.columns if "advantage" in c and c.startswith("away_")]],
        on=["game_id", "club_id"],
        how="left",
        suffixes=("", "_y"),
    )
    cg = cg.drop(columns=[c for c in cg.columns if c.endswith("_y")], errors="ignore")
    for col in ["own_goals", "opponent_goals", "punts"]:
        cg[f"roll_{col}_short"] = cg[f"roll_{col}_home_short"].combine_first(cg[f"roll_{col}_away_short"])
        cg[f"roll_{col}_long"] = cg[f"roll_{col}_home_long"].combine_first(cg[f"roll_{col}_away_long"])

    # ---------- Integració ESPN: map per club_id (Azure o CSV local) ----------
    espn_per_club: Optional[pd.DataFrame] = None
    espn = _load_summary_df("espn_2025_summary", PATH_ESPN_SUMMARY)
    if espn is not None:
        try:
            cols_lower = {c.lower(): c for c in espn.columns}

            def _col(name: str) -> Optional[str]:
                if name in espn.columns:
                    return name
                if name.lower() in cols_lower:
                    return cols_lower[name.lower()]
                return None

            team_id_col = _col("team_id")
            team_name_col = _col("team_name") or team_id_col

            metric_cols: list[str] = []
            for key in [
                "avg_possession",
                "avg_shots_on_target",
                "avg_shots_total",
                "avg_fouls",
                "avg_corners",
                "offensive_index",
                "defensive_index",
                "micro_possession_ewm3",
                "micro_shotsOnTarget_ewm3",
                "micro_totalShots_ewm3",
            ]:
                c = _col(key)
                if c:
                    metric_cols.append(c)

            if team_id_col is not None:
                espn_small = espn[[team_id_col] + metric_cols].drop_duplicates(team_id_col)
                espn_small = espn_small.rename(columns={team_id_col: "team_id"})
                espn_small = espn_small.set_index("team_id")
            else:
                # Fallback: map per nom d'equip amb fuzzy matching (difflib, llindar 0.95)
                espn_small = espn[[team_name_col] + metric_cols].drop_duplicates(team_name_col)
                espn_small["team_norm"] = espn_small[team_name_col].map(
                    lambda x: " ".join(str(x).lower().strip().split())
                )
                clubs_processed["team_norm"] = clubs_processed["name"].map(
                    lambda x: " ".join(str(x).lower().strip().split())
                )
                espn_team_norms = espn_small["team_norm"].dropna().unique().tolist()

                mapping_rows: list[dict[str, Any]] = []
                for _, row in clubs_processed[["club_id", "team_norm"]].iterrows():
                    norm = row["team_norm"]
                    if not isinstance(norm, str) or not norm:
                        continue
                    match = difflib.get_close_matches(norm, espn_team_norms, n=1, cutoff=0.95)
                    if not match:
                        continue
                    best = match[0]
                    espn_row = espn_small[espn_small["team_norm"] == best].iloc[0]
                    data = {"club_id": int(row["club_id"])}
                    for mc in metric_cols:
                        data[mc] = espn_row.get(mc)
                    mapping_rows.append(data)

                if mapping_rows:
                    mapped_df = pd.DataFrame(mapping_rows).set_index("club_id")
                    espn_small = mapped_df
                else:
                    espn_small = None

            if espn_small is not None:
                # Assegurar tipus float32
                for c in espn_small.columns:
                    espn_small[c] = pd.to_numeric(espn_small[c], errors="coerce").astype("float32")
                espn_per_club = espn_small
        except Exception:
            espn_per_club = None

    # ---------- Multi-league 2026 (Azure o CSV local): córners, targetes, market_expectation ----------
    ml_per_club: Optional[pd.DataFrame] = None
    ml_df = _load_summary_df("multi_league_2026_summary", PATH_MULTI_LEAGUE_2026)
    if ml_df is not None and not ml_df.empty:
        try:
            if "club_id" in ml_df.columns and ml_df["club_id"].notna().any():
                cols = ["club_id"]
                for key in ["avg_corners", "avg_yellows", "avg_reds", "market_expectation"]:
                    if key in ml_df.columns:
                        cols.append(key)
                ml_df = ml_df[cols].drop_duplicates("club_id").set_index("club_id")
                ml_df["ml_avg_corners"] = pd.to_numeric(ml_df.get("avg_corners", 0), errors="coerce").fillna(0)
                ml_df["ml_avg_cards"] = (
                    pd.to_numeric(ml_df.get("avg_yellows", 0), errors="coerce").fillna(0)
                    + 2.0 * pd.to_numeric(ml_df.get("avg_reds", 0), errors="coerce").fillna(0)
                )
                ml_df["ml_market_expectation"] = pd.to_numeric(ml_df.get("market_expectation", 0.5), errors="coerce").fillna(0.5)
                ml_per_club = ml_df[["ml_avg_corners", "ml_avg_cards", "ml_market_expectation"]].astype("float32")
        except Exception:
            ml_per_club = None

    # ---------- Per-club features (Home/Away: roll per venue + advantage) ----------
    adv_cols = [c for c in cg.columns if "advantage" in c and (c.startswith("home_") or c.startswith("away_"))]
    roll_home = [c for c in cg.columns if "roll_" in c and "_home_" in c]
    roll_away = [c for c in cg.columns if "roll_" in c and "_away_" in c]
    base_cols = (
        ["game_id", "club_id", "date", "club_market_value", "new_manager_effect", "days_rest"]
        + roll_home
        + roll_away
        + adv_cols
    )
    base_cols = [c for c in base_cols if c in cg.columns]
    per_club = cg[base_cols].drop_duplicates(["game_id", "club_id"])

    if espn_per_club is not None:
        # Flag per saber si un club té dades ESPN específiques (no genèriques)
        per_club["has_espn"] = per_club["club_id"].isin(espn_per_club.index).astype("int8")
        for c in espn_per_club.columns:
            map_col = f"espn_{c}"
            per_club[map_col] = per_club["club_id"].map(espn_per_club[c])
        # Omplir NaN amb mitjana per evitar trencar el model; això també actua com a fallback
        for c in per_club.columns:
            if c.startswith("espn_") and per_club[c].notna().any():
                per_club[c] = per_club[c].fillna(per_club[c].mean())

    if ml_per_club is not None:
        for c in ml_per_club.columns:
            per_club[c] = per_club["club_id"].map(ml_per_club[c])
        for c in ml_per_club.columns:
            if per_club[c].notna().any():
                per_club[c] = per_club[c].fillna(per_club[c].mean())

    df = games[
        [
            "game_id",
            "season",
            "date",
            "competition_id",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        ]
    ].copy()
    df = df.merge(
        per_club,
        left_on=["game_id", "home_club_id"],
        right_on=["game_id", "club_id"],
        how="inner",
    )
    rename_home = {
        "roll_own_goals_home_short": "home_roll_gf_short",
        "roll_opponent_goals_home_short": "home_roll_ga_short",
        "roll_punts_home_short": "home_roll_pts_short",
        "roll_own_goals_home_long": "home_roll_gf_long",
        "roll_opponent_goals_home_long": "home_roll_ga_long",
        "roll_punts_home_long": "home_roll_pts_long",
        "club_market_value": "home_market_value",
        "new_manager_effect": "home_new_manager_effect",
        "days_rest": "home_days_rest",
    }
    for c in per_club.columns:
        if c.startswith("home_advantage"):
            rename_home[c] = c
        if c.startswith("espn_"):
            rename_home[c] = f"home_{c}"
        if c.startswith("ml_"):
            rename_home[c] = f"home_{c}"
    if "has_espn" in per_club.columns:
        rename_home["has_espn"] = "home_has_espn"

    df = df.rename(columns=rename_home)
    df = df.drop(columns=["club_id"], errors="ignore")

    rename_away = {
        "roll_own_goals_away_short": "away_roll_gf_short",
        "roll_opponent_goals_away_short": "away_roll_ga_short",
        "roll_punts_away_short": "away_roll_pts_short",
        "roll_own_goals_away_long": "away_roll_gf_long",
        "roll_opponent_goals_away_long": "away_roll_ga_long",
        "roll_punts_away_long": "away_roll_pts_long",
        "club_market_value": "away_market_value",
        "new_manager_effect": "away_new_manager_effect",
        "days_rest": "away_days_rest",
    }
    for c in per_club.columns:
        if c.startswith("away_advantage"):
            rename_away[c] = c
        if c.startswith("espn_"):
            rename_away[c] = f"away_{c}"
        if c.startswith("ml_"):
            rename_away[c] = f"away_{c}"
    if "has_espn" in per_club.columns:
        rename_away["has_espn"] = "away_has_espn"

    away_feats = per_club.rename(columns=rename_away)
    df = df.merge(
        away_feats,
        left_on=["game_id", "away_club_id"],
        right_on=["game_id", "club_id"],
        how="inner",
    )
    df = df.drop(columns=["club_id"], errors="ignore")

    # Form acceleration: diferència short - long, normalitzat 0-1 (rang aproximat [-3, 3] -> [0, 1])
    df["home_form_acceleration"] = (
        (df["home_roll_pts_short"] - df["home_roll_pts_long"]).fillna(0).add(3).div(6).clip(0, 1).astype("float32")
    )
    df["away_form_acceleration"] = (
        (df["away_roll_pts_short"] - df["away_roll_pts_long"]).fillna(0).add(3).div(6).clip(0, 1).astype("float32")
    )

    # Millora 3: Relative Market Value (proporció local/visitant). Evita divisió per zero.
    df["mv_ratio"] = df["home_market_value"] / df["away_market_value"].clip(lower=1e-6)

    # H2H agregat per parella d'equips (històric): mitjana de gols totals i taxa Over 2.5
    pair_key_games = np.where(
        games["home_club_id"] <= games["away_club_id"],
        games["home_club_id"].astype(str) + "_" + games["away_club_id"].astype(str),
        games["away_club_id"].astype(str) + "_" + games["home_club_id"].astype(str),
    )
    games_h2h = games.copy()
    games_h2h["pair_key"] = pair_key_games
    games_h2h["total_goals"] = games_h2h["home_club_goals"].astype(int) + games_h2h["away_club_goals"].astype(int)
    games_h2h["over25"] = (games_h2h["total_goals"] >= 3).astype(int)
    h2h_group = games_h2h.groupby("pair_key").agg(
        h2h_avg_goals=("total_goals", "mean"),
        h2h_over25_rate=("over25", "mean"),
    )
    h2h_group = h2h_group.astype({"h2h_avg_goals": "float32", "h2h_over25_rate": "float32"})

    pair_key_df = np.where(
        df["home_club_id"] <= df["away_club_id"],
        df["home_club_id"].astype(str) + "_" + df["away_club_id"].astype(str),
        df["away_club_id"].astype(str) + "_" + df["home_club_id"].astype(str),
    )
    df["pair_key"] = pair_key_df
    df = df.merge(h2h_group, on="pair_key", how="left")
    df = df.drop(columns=["pair_key"], errors="ignore")

    # StatsBomb (Azure o CSV local): relative_xg_strength i tactical_danger_index
    sb_avg_xg_map: dict[tuple[str, int], float] = {}
    sb_tactical_danger_map: dict[tuple[str, int], float] = {}
    sb_df = _load_summary_df("statsbomb_summary", PATH_STATSBOMB_SUMMARY)
    if sb_df is not None and not sb_df.empty:
        try:
            for c in ["team_name", "season_year", "avg_xg"]:
                if c not in sb_df.columns:
                    raise ValueError(f"Falta columna {c}")
            sb_df["season_year"] = pd.to_numeric(sb_df["season_year"], errors="coerce").fillna(0).astype(int)
            sb_df["avg_xg"] = pd.to_numeric(sb_df["avg_xg"], errors="coerce").fillna(0.0)
            if "avg_key_passes" in sb_df.columns:
                sb_df["avg_key_passes"] = pd.to_numeric(sb_df["avg_key_passes"], errors="coerce").fillna(0.0)
            else:
                sb_df["avg_key_passes"] = 0.0
            sb_df["team_norm"] = sb_df["team_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            sb_df["tactical_danger_index"] = (sb_df["avg_xg"] * sb_df["avg_key_passes"]) / 100.0
            for _, row in sb_df.iterrows():
                key = (row["team_norm"], int(row["season_year"]))
                sb_avg_xg_map[key] = float(row["avg_xg"])
                sb_tactical_danger_map[key] = float(row["tactical_danger_index"])
        except Exception:
            sb_avg_xg_map = {}
            sb_tactical_danger_map = {}

    def _norm(s: str) -> str:
        return " ".join(str(s).lower().strip().split()) if pd.notna(s) and str(s).strip() else ""

    if sb_avg_xg_map:
        club_name = clubs.set_index("club_id")["name"]
        df["_home_norm"] = df["home_club_id"].map(club_name).fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        df["_away_norm"] = df["away_club_id"].map(club_name).fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        sb_lookup = pd.DataFrame(
            [(*k, v) for k, v in sb_avg_xg_map.items()],
            columns=["team_norm", "season_year", "avg_xg"],
        ).drop_duplicates(subset=["team_norm", "season_year"], keep="first")
        df = df.merge(
            sb_lookup.rename(columns={"team_norm": "_home_norm", "season_year": "season", "avg_xg": "_home_sb_xg"}),
            on=["_home_norm", "season"],
            how="left",
            suffixes=("", "_sb"),
        )
        if "_home_sb_xg" not in df.columns:
            df["_home_sb_xg"] = np.nan
        df = df.merge(
            sb_lookup.rename(columns={"team_norm": "_away_norm", "season_year": "season", "avg_xg": "_away_sb_xg"}),
            on=["_away_norm", "season"],
            how="left",
        )
        if "_away_sb_xg" not in df.columns:
            df["_away_sb_xg"] = np.nan
        df["relative_xg_strength"] = (df["_home_sb_xg"] / df["_away_sb_xg"].clip(lower=1e-6)).fillna(1.0).clip(0.2, 5.0).astype("float32")
        df["_has_sb"] = (df["relative_xg_strength"] != 1.0).astype(np.int8)
        df["relative_xg_strength"] = ((df["relative_xg_strength"] - 0.2) / 4.8).clip(0, 1).astype("float32")
        df = df.drop(columns=["_home_sb_xg", "_away_sb_xg"], errors="ignore")
        # tactical_danger_index per equip (StatsBomb Top 10)
        if sb_tactical_danger_map:
            td_lookup = pd.DataFrame(
                [(*k, v) for k, v in sb_tactical_danger_map.items()],
                columns=["team_norm", "season_year", "tactical_danger_index"],
            ).drop_duplicates(subset=["team_norm", "season_year"], keep="first")
            df = df.merge(
                td_lookup.rename(columns={"team_norm": "_home_norm", "season_year": "season", "tactical_danger_index": "home_tactical_danger_index"}),
                on=["_home_norm", "season"],
                how="left",
            )
            df = df.merge(
                td_lookup.rename(columns={"team_norm": "_away_norm", "season_year": "season", "tactical_danger_index": "away_tactical_danger_index"}),
                on=["_away_norm", "season"],
                how="left",
            )
            df["home_tactical_danger_index"] = df["home_tactical_danger_index"].fillna(0.0).astype("float32")
            df["away_tactical_danger_index"] = df["away_tactical_danger_index"].fillna(0.0).astype("float32")
        else:
            df["home_tactical_danger_index"] = 0.0
            df["away_tactical_danger_index"] = 0.0
        df = df.drop(columns=["_home_norm", "_away_norm"], errors="ignore")
    else:
        df["relative_xg_strength"] = (1.0 - 0.2) / 4.8
        df["_has_sb"] = 0
        df["home_tactical_danger_index"] = 0.0
        df["away_tactical_danger_index"] = 0.0

    # Variable competition_type (Lliga vs Copa vs Champions) des de competitions (Azure o CSV)
    comp_type_map: dict[str, str] = {}
    comps: Optional[pd.DataFrame] = None

    # Si hi ha Azure (st.secrets), només consultem la taula 'competitions'
    try:
        import streamlit as st  # només per comprovar secrets

        s = getattr(st, "secrets", None)
        has_azure = bool(s and s.get("azure_sql"))
    except Exception:
        has_azure = False

    if has_azure:
        comps = get_data_from_azure("competitions")
    elif os.path.isfile(PATH_COMPETITIONS):
        try:
            comps = pd.read_csv(
                PATH_COMPETITIONS,
                usecols=["competition_id", "sub_type", "type"],
                dtype={"competition_id": "string", "sub_type": "string", "type": "string"},
                low_memory=False,
            )
        except Exception:
            comps = None

    if comps is not None and not comps.empty:
        def _map_comp(row: pd.Series) -> str:
            t = (row.get("type") or "").lower()
            st = (row.get("sub_type") or "").lower()
            if "uefa_champions_league" in st:
                return "Champions"
            if t == "domestic_league":
                return "League"
            if t == "international_cup" and ("uefa" in st or "europa" in st or "conference" in st):
                return "Champions"
            return "Cup"

        comps["competition_type"] = comps.apply(_map_comp, axis=1)
        comp_type_map = comps.set_index("competition_id")["competition_type"].to_dict()

    df["competition_type"] = df["competition_id"].map(comp_type_map).fillna("League")
    comp_code_map = {"League": 0, "Cup": 1, "Champions": 2}
    df["competition_type_code"] = df["competition_type"].map(comp_code_map).fillna(0).astype("int8")

    # Target Over 2.5: 1 si total gols >= 3, 0 en cas contrari
    df["over_25"] = ((df["home_club_goals"].astype(int) + df["away_club_goals"].astype(int)) >= 3).astype(np.int8)

    # Normalització per lliga: mitjana de gols per competició (Bundes vs Serie A, etc.)
    # Separa estils: lligues amb més gols (Bundes) vs menys (Serie A); valor 0-1.
    df["_total_goals"] = df["home_club_goals"].astype(int) + df["away_club_goals"].astype(int)
    league_avg = df.groupby("competition_id")["_total_goals"].transform("mean")
    df["relative_league_goals"] = (df["_total_goals"] / league_avg.clip(lower=1e-6)).clip(0.5, 2.0).astype("float32")
    df["relative_league_goals"] = ((df["relative_league_goals"] - 0.5) / 1.5).clip(0, 1).astype("float32")
    df = df.drop(columns=["_total_goals"], errors="ignore")

    # Filtre temporal dràstic: només temporades >= 2018 (futbol anterior = soroll per a 2026)
    df = df[df["season"] >= MIN_SEASON].copy()

    # Eliminatòria (knockout): per a dades històriques, per defecte 0 (majoria són lliga)
    df["is_knockout"] = 0
    df["is_return_leg"] = 0
    df["first_leg_diff"] = 0.0

    # Selecció de feature_cols (ESPN, H2H, eliminatòria, etc.). Exclusió de quotes individuals.
    feat_cols = [
        c
        for c in df.columns
        if (
            (c.startswith("home_roll_")
            or c.startswith("away_roll_")
            or c.startswith("home_advantage")
            or c.startswith("away_advantage")
            or c.startswith("home_market")
            or c.startswith("away_market")
            or c.startswith("home_new_manager")
            or c.startswith("away_new_manager")
            or c.startswith("home_espn_")
            or c.startswith("away_espn_")
            or c.startswith("home_ml_")
            or c.startswith("away_ml_")
            or c in ("h2h_avg_goals", "h2h_over25_rate")
            or c
            in (
                "mv_ratio",
                "home_days_rest",
                "away_days_rest",
                "relative_xg_strength",
                "home_form_acceleration",
                "away_form_acceleration",
                "home_tactical_danger_index",
                "away_tactical_danger_index",
                "is_knockout",
                "is_return_leg",
                "first_leg_diff",
                "relative_league_goals",
            ))
            and not any(odds in c for odds in ODDS_INDIVIDUAL_BLACKLIST)
        )
    ]
    for col in feat_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    return df, feat_cols, cg, clubs_processed, meta


H2H_LAST_N = 5


def entrenar_model(df: pd.DataFrame, feature_cols_list: list[str]) -> tuple[Any, float, pd.DataFrame, np.ndarray]:
    """Entrenament XGBoost binari (Over 2.5) amb split cronològic. Retorna model, accuracy, importància, matriu de confusió."""
    train_df = df[df["season"] <= TRAIN_SEASON_MAX].copy()
    test_df = df[df["season"] == TEST_SEASON].copy()

    if test_df.empty:
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        df_sorted = df_copy.dropna(subset=["date"]).sort_values("date")
        n = len(df_sorted)
        train_df = df_sorted.iloc[: int(0.8 * n)]
        test_df = df_sorted.iloc[int(0.8 * n) :]

    # H2H pes real (0.7): històric directe important; h2h_over25_rate extremes (>0.8 o <0.2) s’apropen a 0.5
    train_df = train_df.copy()
    test_df = test_df.copy()
    def _cap_h2h(df: pd.DataFrame) -> None:
        if "h2h_over25_rate" in df.columns:
            r = df["h2h_over25_rate"].astype(float)
            mask_extreme = (r > 0.8) | (r < 0.2)
            df["h2h_over25_rate"] = np.where(mask_extreme, 0.5 + (r - 0.5) * 0.3, r)
        for col in ("h2h_avg_goals", "h2h_over25_rate"):
            if col in df.columns:
                df[col] = df[col] * 0.7
    _cap_h2h(train_df)
    _cap_h2h(test_df)

    X_train = train_df[feature_cols_list].astype(float)
    y_train = train_df["over_25"].astype(int)
    X_test = test_df[feature_cols_list].astype(float)
    y_test = test_df["over_25"].astype(int)

    # interaction_constraints: vitalitat (StatsBomb + dinàmiques + ESPN) en un grup, H2H en l’altre
    # XGBoost espera llistes de noms de columnes (coincidents amb X_train.columns)
    h2h_names = {"h2h_avg_goals", "h2h_over25_rate"}
    vitality_names = [c for c in feature_cols_list if c not in h2h_names]
    h2h_list = [c for c in feature_cols_list if c in h2h_names]
    interaction_constraints = [vitality_names, h2h_list] if vitality_names and h2h_list else None

    clf = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=5,
        learning_rate=0.01,
        colsample_bytree=0.8,
        reg_lambda=15.0,
        min_child_weight=5,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        interaction_constraints=interaction_constraints,
    )

    # Ponderació temporal: pes 2.0 (2024-2026), 1.0 (2020-2023), 0.5 (resta)
    sample_weight = np.ones(len(X_train), dtype="float32")
    if "season" in train_df.columns:
        s = train_df["season"].to_numpy()
        sample_weight = np.where(s >= 2024, 2.0, sample_weight)
        sample_weight = np.where((s >= 2020) & (s < 2024), 1.0, sample_weight)
        sample_weight = np.where(s < 2020, 0.5, sample_weight)

    # Entrenament amb conjunt de validació; la teva versió de xgboost no admet
    # early_stopping_rounds des de fit(), així que el deixem només com a eval_set.
    clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0.0
    conf_mat = confusion_matrix(y_test, y_pred) if len(y_test) > 0 else np.zeros((2, 2))
    if len(y_test) > 0:
        print(classification_report(y_test, y_pred, target_names=["Under 2.5", "Over 2.5"]))

    imp = pd.DataFrame({
        "variable": feature_cols_list,
        "importancia": clf.feature_importances_,
    }).sort_values("importancia", ascending=False)
    print("\n--- Top 20 variables més importants ---")
    print(imp.head(20).to_string(index=False))

    # Variables altament correlacionades: per a cada parella amb corr > 0.9, marcar la de menor importància
    imp_map = dict(zip(imp["variable"], imp["importancia"]))
    try:
        corr = X_train.corr()
        to_drop: list[str] = []
        for i, col_i in enumerate(feature_cols_list):
            for col_j in feature_cols_list[i + 1 :]:
                if col_i not in corr.columns or col_j not in corr.columns:
                    continue
                r = corr.loc[col_i, col_j]
                if abs(r) > 0.9:
                    drop = col_j if imp_map.get(col_i, 0) >= imp_map.get(col_j, 0) else col_i
                    if drop not in to_drop:
                        to_drop.append(drop)
        if to_drop:
            os.makedirs(CSV_DIR, exist_ok=True)
            with open(PATH_FEATURES_CORRELATED_DROP, "w", encoding="utf-8") as f:
                for v in to_drop:
                    f.write(v + "\n")
            print(f"Variables correlacionades (drop): {len(to_drop)} → {PATH_FEATURES_CORRELATED_DROP}")
    except Exception as e:
        print(f"Correlacions no guardades: {e}")

    return clf, acc, imp, conf_mat


def poisson_score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 6,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """
    Matriu de probabilitats de marcador (Poisson independent) del 0-0 al (max_goals-1)-(max_goals-1).
    lambda_home i lambda_away són els gols esperats (p. ex. EWM de gols a favor ajustats per valor).
    Retorna: (matriu 6x6, llista dels 3 marcadors amb més probabilitat [( "2-1", 12.3 ), ...]).
    """
    lambda_home = max(0.01, float(lambda_home))
    lambda_away = max(0.01, float(lambda_away))
    mat = np.zeros((max_goals, max_goals))
    scores_list: list[tuple[int, int, float]] = []
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            mat[i, j] = p
            scores_list.append((i, j, p))
    total = mat.sum()
    if total > 0:
        mat = mat / total
        scores_list = [(i, j, (p / total) * 100) for i, j, p in scores_list]
    scores_list.sort(key=lambda x: -x[2])
    top3 = [(f"{i}-{j}", round(p, 1)) for i, j, p in scores_list[:3]]
    return mat, top3


def probabilitats_gols_des_de_poisson(mat: np.ndarray) -> tuple[float, float, float]:
    """
    A partir de la matriu de Poisson (mat[i,j] = P(local=i, visitant=j)), retorna
    (over_05_pct, over_15_pct, over_25_pct) en percentatge 0-100.
    Over 0.5 = 1 - P(0-0), Over 1.5 = 1 - P(total<=1), Over 2.5 = 1 - P(total<=2).
    """
    mat = np.asarray(mat)
    p_00 = float(mat[0, 0])
    p_total_1 = p_00 + (mat[1, 0] if mat.shape[0] > 1 else 0) + (mat[0, 1] if mat.shape[1] > 1 else 0)
    p_total_2 = p_total_1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i + j == 2:
                p_total_2 += float(mat[i, j])
    over_05 = (1.0 - p_00) * 100
    over_15 = (1.0 - p_total_1) * 100
    over_25 = (1.0 - p_total_2) * 100
    return max(0, min(100, over_05)), max(0, min(100, over_15)), max(0, min(100, over_25))


def probabilitats_1x2_des_de_poisson(mat: np.ndarray) -> tuple[float, float, float]:
    """
    A partir de la matriu de Poisson, retorna (p_1, p_x, p_2) en percentatge: victòria local, empat, victòria visitant.
    """
    mat = np.asarray(mat)
    p_1 = float(np.sum(np.tril(mat, -1))) * 100   # i > j
    p_x = float(np.sum(np.diag(mat))) * 100       # i == j
    p_2 = float(np.sum(np.triu(mat, 1))) * 100   # j > i
    total = p_1 + p_x + p_2
    if total > 0:
        p_1, p_x, p_2 = p_1 / total * 100, p_x / total * 100, p_2 / total * 100
    return p_1, p_x, p_2


def cercar_equip(nom: str, clubs_df_ref: pd.DataFrame) -> tuple[Optional[int], list[str]]:
    """
    Cerca un equip per nom (exacte, parcial o per paraules).
    Retorna (club_id, suggeriments) o (None, suggeriments). Fuzzy matching per l'app.
    """
    nom = str(nom).strip() if nom is not None else ""
    if not nom:
        return None, []
    clubs = clubs_df_ref[["club_id", "name"]].drop_duplicates("club_id")
    name_lower = clubs["name"].str.lower()
    nom_lower = nom.lower()

    exact = clubs[name_lower == nom_lower]
    if len(exact) > 0:
        return int(exact.iloc[0]["club_id"]), [exact.iloc[0]["name"]]

    partial = clubs[name_lower.str.contains(nom_lower, na=False)]
    if len(partial) == 1:
        return int(partial.iloc[0]["club_id"]), [partial.iloc[0]["name"]]
    if len(partial) > 1:
        partial = partial.copy()
        partial["_len"] = partial["name"].str.len()
        partial = partial.sort_values("_len").drop(columns=["_len"])
        return int(partial.iloc[0]["club_id"]), partial["name"].tolist()[:5]

    paraules = [p for p in nom_lower.split() if len(p) > 1]
    if paraules:
        mascara = pd.Series(True, index=clubs.index)
        for p in paraules:
            mascara &= name_lower.str.contains(p, na=False)
        by_words = clubs[mascara]
        if len(by_words) == 1:
            return int(by_words.iloc[0]["club_id"]), [by_words.iloc[0]["name"]]
        if len(by_words) > 1:
            by_words = by_words.copy()
            by_words["_len"] = by_words["name"].str.len()
            by_words = by_words.sort_values("_len").drop(columns=["_len"])
            return int(by_words.iloc[0]["club_id"]), by_words["name"].tolist()[:5]

    starts = clubs[name_lower.str.startswith(nom_lower, na=False)]
    if len(starts) > 0:
        return None, starts["name"].tolist()[:5]
    return None, []


def get_h2h_matches(id_home: Optional[int], id_away: Optional[int], games_df: Optional[pd.DataFrame], max_n: int = 3) -> list[dict[str, Any]]:
    """
    Retorna els últims max_n enfrontaments directes entre id_home i id_away.
    Filtre Pandas directe per home_club_id / away_club_id (sense iterar tot el dataset).
    Cada element: {"season": int, "score": "2-1", "total_goals": 3, "over_25": True}.
    """
    if id_home is None or id_away is None or games_df is None or games_df.empty:
        return []
    g = games_df.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"])
    # Filtre directe: (local=A, visitant=B) o (local=B, visitant=A)
    mask = (
        ((g["home_club_id"] == id_home) & (g["away_club_id"] == id_away))
        | ((g["home_club_id"] == id_away) & (g["away_club_id"] == id_home))
    )
    h2h = g.loc[mask].sort_values("date", ascending=False).head(max_n)
    out = []
    for _, row in h2h.iterrows():
        hg = int(row["home_club_goals"])
        ag = int(row["away_club_goals"])
        if row["home_club_id"] == id_away and row["away_club_id"] == id_home:
            hg, ag = ag, hg
        out.append({
            "season": int(row.get("season", 0)),
            "score": f"{hg}-{ag}",
            "total_goals": hg + ag,
            "over_25": (hg + ag) >= 3,
        })
    return out


def predictor(
    nom_local: str,
    nom_visitant: str,
    verbose: bool = True,
    is_knockout: bool = False,
    is_return_leg: bool = False,
    first_leg_diff: float = 0.0,
) -> Optional[dict[str, Any]]:
    """
    Predicció Over 2.5 gols (probabilitat en %) a partir dels noms de dos equips.
    is_knockout / is_return_leg / first_leg_diff: opcions d'eliminatòria (tornada, diferència anada).
    Retorna over_25_prob, h2h_matches, score_matrix, etc.
    """
    global model, feature_cols, clubs_df, games_full

    if model is None or clubs_df is None or games_full is None:
        if verbose:
            print("Primer cal cridar main() o carregar el model (p. ex. des de l'app).")
        return None
    if not feature_cols:
        if verbose:
            print("La llista de features del model és buida. Cal reentrenar (main()).")
        return None

    id_local, sugg_local = cercar_equip(nom_local, clubs_df)
    id_away, sugg_away = cercar_equip(nom_visitant, clubs_df)

    # Si un equip no es troba al CSV, no bloquegem: fem servir mitjana global (fallback)
    generic_team_local = id_local is None
    generic_team_away = id_away is None
    if id_local is None:
        if verbose:
            print(f"Equip no trobat al dataset: '{nom_local}'. S'usen dades genèriques (mitjana global).")
            if sugg_local:
                print("  Suggeriments:", ", ".join(sugg_local))
    if id_away is None:
        if verbose:
            print(f"Equip no trobat al dataset: '{nom_visitant}'. S'usen dades genèriques (mitjana global).")
            if sugg_away:
                print("  Suggeriments:", ", ".join(sugg_away))

    last_stats = games_full["last_stats_per_club"]
    last_match_date_per_club = games_full.get("last_match_date_per_club")
    mean_days_rest = float(games_full.get("mean_days_rest", 7.0))
    mean_gf = games_full["mean_roll_gf"]
    mean_ga = games_full["mean_roll_ga"]
    mean_pts = games_full["mean_roll_pts"]
    mean_mv = games_full["mean_market_value"]
    mean_new_mgr = games_full.get("mean_new_manager_effect", 0.0)
    espn_per_club = games_full.get("espn_per_club")
    espn_global_means = games_full.get("espn_global_means", {})
    ml_per_club = games_full.get("ml_per_club")
    ml_global_means = games_full.get("ml_global_means", {})

    today = date.today()
    def _days_rest(club_id: Optional[int]) -> float:
        if club_id is None or last_match_date_per_club is None or club_id not in last_match_date_per_club.index:
            return mean_days_rest
        last_d = last_match_date_per_club[club_id]
        if pd.isna(last_d):
            return mean_days_rest
        d = last_d.date() if hasattr(last_d, "date") else last_d
        delta = (today - d).days
        return max(0, float(delta))

    home_days_rest = _days_rest(id_local)
    away_days_rest = _days_rest(id_away)

    row_home = last_stats[last_stats["club_id"] == id_local] if id_local is not None else pd.DataFrame()
    row_away = last_stats[last_stats["club_id"] == id_away] if id_away is not None else pd.DataFrame()

    def _get_roll(r: pd.Series, prefix: str, mean_gf: float, mean_ga: float, mean_pts: float) -> tuple:
        # Preferència per columnes per camp (home/away); fallback a genèriques o mitjana
        def _v(spec: str, gen: str, fallback: float) -> float:
            val = r.get(spec)
            if pd.notna(val):
                return float(val)
            if gen in r.index and pd.notna(r.get(gen)):
                return float(r[gen])
            return fallback
        gf_s = _v(f"roll_own_goals_{prefix}_short", "roll_own_goals_short", mean_gf)
        ga_s = _v(f"roll_opponent_goals_{prefix}_short", "roll_opponent_goals_short", mean_ga)
        pts_s = _v(f"roll_punts_{prefix}_short", "roll_punts_short", mean_pts)
        gf_l = _v(f"roll_own_goals_{prefix}_long", "roll_own_goals_long", mean_gf)
        ga_l = _v(f"roll_opponent_goals_{prefix}_long", "roll_opponent_goals_long", mean_ga)
        pts_l = _v(f"roll_punts_{prefix}_long", "roll_punts_long", mean_pts)
        return gf_s, ga_s, pts_s, gf_l, ga_l, pts_l

    if len(row_home) == 0:
        home_gf_short = home_gf_long = mean_gf
        home_ga_short = home_ga_long = mean_ga
        home_pts_short = home_pts_long = mean_pts
        home_mv = mean_mv
        home_new_mgr_eff = mean_new_mgr
        home_adv = {}
    else:
        r = row_home.iloc[0]
        home_gf_short, home_ga_short, home_pts_short, home_gf_long, home_ga_long, home_pts_long = _get_roll(r, "home", mean_gf, mean_ga, mean_pts)
        home_mv = float(r["club_market_value"]) if pd.notna(r.get("club_market_value")) else mean_mv
        home_new_mgr_eff = float(r.get("new_manager_effect", 0) or 0)
        home_adv = {c: float(r[c]) for c in row_home.columns if str(c).startswith("home_advantage") and pd.notna(r.get(c))}

    if len(row_away) == 0:
        away_gf_short = away_gf_long = mean_gf
        away_ga_short = away_ga_long = mean_ga
        away_pts_short = away_pts_long = mean_pts
        away_mv = mean_mv
        away_new_mgr_eff = mean_new_mgr
        away_adv = {}
    else:
        r = row_away.iloc[0]
        away_gf_short, away_ga_short, away_pts_short, away_gf_long, away_ga_long, away_pts_long = _get_roll(r, "away", mean_gf, mean_ga, mean_pts)
        away_mv = float(r["club_market_value"]) if pd.notna(r.get("club_market_value")) else mean_mv
        away_new_mgr_eff = float(r.get("new_manager_effect", 0) or 0)
        away_adv = {c: float(r[c]) for c in row_away.columns if str(c).startswith("away_advantage") and pd.notna(r.get(c))}

    mv_ratio = home_mv / max(away_mv, 1e-6)

    # Micro-estadístiques ESPN per a cada equip (si disponibles)
    def _espn_stats(club_id: Optional[int]) -> tuple[dict[str, float], bool]:
        default = {k: float(v) for k, v in espn_global_means.items()}
        if espn_per_club is None or club_id is None or club_id not in espn_per_club.index:
            return default, False
        row_club = espn_per_club.loc[club_id]
        out = {}
        for k in default.keys():
            col = f"espn_{k}" if f"espn_{k}" in row_club.index else k
            val = row_club.get(col, np.nan)
            out[k] = float(val) if pd.notna(val) else default[k]
        return out, True

    espn_home, home_espn_has_data = _espn_stats(id_local)
    espn_away, away_espn_has_data = _espn_stats(id_away)

    # H2H on-demand: només per als dos equips seleccionats (filtre Pandas directe)
    games_df = games_full.get("games_df")
    league_avg_goals = float(games_full.get("league_avg_goals", 2.7))
    league_over25_rate = float(games_full.get("league_over25_rate", 0.52))
    h2h_matches = get_h2h_matches(id_local, id_away, games_df, max_n=H2H_LAST_N)
    # (h2h_avg/h2h_rate no s'usen al model; el model s'entrena sense H2H per evitar preprocessament pesat)

    # Construir X amb el mateix ordre de columnes que a l'entrenament (feature_cols)
    home_form_acc = home_pts_short - home_pts_long
    away_form_acc = away_pts_short - away_pts_long
    sb_tactical = games_full.get("sb_tactical_danger_per_club") or {}
    home_td = float(sb_tactical.get(id_local, 0.0)) if id_local else 0.0
    away_td = float(sb_tactical.get(id_away, 0.0)) if id_away else 0.0

    row = {
        "home_roll_gf_short": home_gf_short,
        "home_roll_ga_short": home_ga_short,
        "home_roll_pts_short": home_pts_short,
        "home_roll_gf_long": home_gf_long,
        "home_roll_ga_long": home_ga_long,
        "home_roll_pts_long": home_pts_long,
        "away_roll_gf_short": away_gf_short,
        "away_roll_ga_short": away_ga_short,
        "away_roll_pts_short": away_pts_short,
        "away_roll_gf_long": away_gf_long,
        "away_roll_ga_long": away_ga_long,
        "away_roll_pts_long": away_pts_long,
        "home_form_acceleration": home_form_acc,
        "away_form_acceleration": away_form_acc,
        "home_market_value": home_mv,
        "away_market_value": away_mv,
        "home_new_manager_effect": home_new_mgr_eff,
        "away_new_manager_effect": away_new_mgr_eff,
        "mv_ratio": mv_ratio,
        "home_days_rest": home_days_rest,
        "away_days_rest": away_days_rest,
        "home_has_espn": int(home_espn_has_data),
        "away_has_espn": int(away_espn_has_data),
        "home_tactical_danger_index": home_td,
        "away_tactical_danger_index": away_td,
        "is_knockout": 1 if is_knockout else 0,
        "is_return_leg": 1 if is_return_leg else 0,
        "first_leg_diff": float(first_leg_diff),
    }
    for k, v in home_adv.items():
        row[k] = v
    for k, v in away_adv.items():
        row[k] = v
    for k, v in espn_home.items():
        row[f"home_espn_{k}"] = v
    for k, v in espn_away.items():
        row[f"away_espn_{k}"] = v

    # Multi-league 2026 (córners, targetes, market_expectation) per al model
    def _ml_stats(club_id: Optional[int]) -> dict[str, float]:
        out = {k: float(v) for k, v in ml_global_means.items()}
        if ml_per_club is not None and club_id is not None and club_id in ml_per_club.index:
            for c in ml_per_club.columns:
                out[c] = float(ml_per_club.loc[club_id, c])
        return out

    ml_home = _ml_stats(id_local)
    ml_away = _ml_stats(id_away)
    for k, v in ml_home.items():
        row[f"home_{k}"] = v
    for k, v in ml_away.items():
        row[f"away_{k}"] = v

    if h2h_matches:
        h2h_avg = sum(m["total_goals"] for m in h2h_matches) / len(h2h_matches)
        h2h_rate = sum(1 for m in h2h_matches if m["over_25"]) / len(h2h_matches)
    else:
        h2h_avg = league_avg_goals
        h2h_rate = league_over25_rate
    if h2h_rate > 0.8 or h2h_rate < 0.2:
        h2h_rate = 0.5 + (h2h_rate - 0.5) * 0.3
    row["h2h_avg_goals"] = h2h_avg * 0.7
    row["h2h_over25_rate"] = h2h_rate * 0.7

    # relative_xg_strength (StatsBomb): estil de joc històric; normalitzat 0-1 (mateix que a l'entrenament)
    sb_per_club = games_full.get("sb_per_club") or {}
    home_sb = float(sb_per_club.get(id_local, 1.0)) if id_local else 1.0
    away_sb = float(sb_per_club.get(id_away, 1.0)) if id_away else 1.0
    raw_xg = float(np.clip(home_sb / max(away_sb, 1e-6), 0.2, 5.0))
    row["relative_xg_strength"] = float(np.clip((raw_xg - 0.2) / 4.8, 0, 1))

    # relative_league_goals: en predicció sense competició concreta, neutral (0.5 = mitjana)
    row["relative_league_goals"] = 0.5

    # Form acceleration: mateixa normalització 0-1 que a l'entrenament (rang [-3, 3] -> [0, 1])
    row["home_form_acceleration"] = float(np.clip((row["home_form_acceleration"] + 3) / 6, 0, 1))
    row["away_form_acceleration"] = float(np.clip((row["away_form_acceleration"] + 3) / 6, 0, 1))

    values = [float(row.get(c, 0)) for c in feature_cols]
    X = pd.DataFrame([values], columns=feature_cols).astype(float)
    probs = model.predict_proba(X)[0]
    over_25_prob = float(probs[1]) * 100
    # Tornada amb desavantatge: l’equip local ha d’arriscar → augmentar lleugerament P(Over 2.5)
    if is_return_leg and first_leg_diff <= -2:
        over_25_prob = min(95, over_25_prob + 5.0)
    result: dict[str, Any] = {"over_25_prob": over_25_prob}

    nom_canon_local = str(clubs_df[clubs_df["club_id"] == id_local]["name"].iloc[0]) if id_local is not None else str(nom_local)
    nom_canon_away = str(clubs_df[clubs_df["club_id"] == id_away]["name"].iloc[0]) if id_away is not None else str(nom_visitant)

    # Gols esperats (Poisson): EWM de gols a favor lleugerament ajustats pel valor relatiu
    log_ratio = np.log(mv_ratio) if mv_ratio > 0 else 0
    lambda_home = max(0.05, home_gf_short * (1 + 0.12 * np.clip(log_ratio, -1.5, 1.5)))
    lambda_away = max(0.05, away_gf_short * (1 - 0.12 * np.clip(log_ratio, -1.5, 1.5)))
    score_mat, top3_exact = poisson_score_matrix(lambda_home, lambda_away, max_goals=6)

    # Multi-mercat gols: Over 0.5 i Over 1.5 des de Poisson; Over 2.5 del model (accuracy 57.2%)
    over_05_pct, over_15_pct, _ = probabilitats_gols_des_de_poisson(score_mat)

    # Blend XGBoost + Poisson per Over 2.5 i imposar monotonia:
    # P(+2.5) <= P(+1.5) <= P(+0.5) i cap probabilitat 0%/100%.
    over_25_blend = 0.7 * over_25_prob + 0.3 * over_15_pct
    # Clamp visual
    over_05_clamped = max(5.0, min(99.0, over_05_pct))
    over_15_clamped = max(5.0, min(over_05_clamped, over_15_pct))
    over_25_clamped = max(5.0, min(over_15_clamped - 1.0, over_25_blend))

    result["over_05_prob"] = over_05_clamped
    result["over_15_prob"] = over_15_clamped
    result["over_25_prob"] = over_25_clamped

    # 1X2 des de la matriu de Poisson (coherent amb el marcador exacte)
    p_1, p_x, p_2 = probabilitats_1x2_des_de_poisson(score_mat)
    result["p_1"] = p_1
    result["p_x"] = p_x
    result["p_2"] = p_2

    result["top3_exact"] = top3_exact
    result["score_matrix"] = score_mat
    result["lambda_home"] = lambda_home
    result["lambda_away"] = lambda_away
    result["home_mv"] = home_mv
    result["away_mv"] = away_mv
    result["home_roll_gf"] = home_gf_short
    result["away_roll_gf"] = away_gf_short
    result["home_roll_ga"] = home_ga_short
    result["away_roll_ga"] = away_ga_short
    result["home_roll_gf_short"] = home_gf_short
    result["home_roll_gf_long"] = home_gf_long
    result["away_roll_gf_short"] = away_gf_short
    result["away_roll_gf_long"] = away_gf_long
    result["home_form_acceleration"] = home_form_acc
    result["away_form_acceleration"] = away_form_acc
    # Exposar micro-variables ESPN per a la UI
    result["home_espn_avg_possession"] = float(espn_home.get("avg_possession", 0.0))
    result["home_espn_avg_shots_on_target"] = float(espn_home.get("avg_shots_on_target", 0.0))
    result["away_espn_avg_possession"] = float(espn_away.get("avg_possession", 0.0))
    result["away_espn_avg_shots_on_target"] = float(espn_away.get("avg_shots_on_target", 0.0))
    result["home_espn_has_data"] = home_espn_has_data
    result["away_espn_has_data"] = away_espn_has_data
    result["home_corners_metric"] = float(ml_home.get("ml_avg_corners", 0))
    result["away_corners_metric"] = float(ml_away.get("ml_avg_corners", 0))
    result["home_cards_metric"] = float(ml_home.get("ml_avg_cards", 0))
    result["away_cards_metric"] = float(ml_away.get("ml_avg_cards", 0))
    result["nom_local"] = nom_canon_local
    result["nom_visitant"] = nom_canon_away
    result["id_local"] = id_local
    result["id_away"] = id_away
    result["generic_team_local"] = generic_team_local
    result["generic_team_away"] = generic_team_away
    result["home_days_rest"] = int(home_days_rest)
    result["away_days_rest"] = int(away_days_rest)
    result["h2h_matches"] = h2h_matches

    if verbose:
        print(f"\nPredicció: {nom_canon_local} vs {nom_canon_away}")
        print(f"  P(Over 2.5) = {result['over_25_prob']:.1f}%")
        print("  Marcadors exactes (top 3): " + ", ".join(f"{s} ({p}%)" for s, p in top3_exact))

    return result


def _normalize_name(name: str) -> str:
    return " ".join(str(name).lower().strip().split())


def _is_key_player_missing(injury_name: str, key_player_names: list[str]) -> Optional[str]:
    """Retorna el nom del jugador clau que coincideix amb la baixa, o None."""
    inj = _normalize_name(injury_name)
    for kp in key_player_names:
        kp_n = _normalize_name(kp)
        if kp_n in inj or inj in kp_n or kp_n.replace(" ", "") in inj.replace(" ", ""):
            return kp
    return None


def _key_players_absent_from_squad(
    key_player_names: list[str],
    available_names: list[str],
) -> list[str]:
    """
    Retorna els jugadors clau que NO apareixen a la llista de disponibles (lineup/squad API).
    Comparació per similitud de nom (fuzzy).
    """
    available_norm = {_normalize_name(n) for n in available_names}
    absent = []
    for kp in key_player_names:
        kp_n = _normalize_name(kp)
        found = any(
            kp_n in av or av in kp_n or kp_n.replace(" ", "") in av.replace(" ", "")
            for av in available_norm
        )
        if not found:
            absent.append(kp)
    return absent


# Penalitzador per baixa de jugador clau: 12% per jugador, màxim 5 (60% de reducció)
PENALTY_PER_KEY_MISSING = 0.12
MAX_KEY_PLAYERS_PENALIZED = 5


def apply_live_adjustment(
    result: dict[str, Any],
    live_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Ajusta la predicció amb dades en viu (API): baixes de jugadors clau (especialment davanters)
    penalitzen la probabilitat d'Over 2.5. Retorna result amb over_25_prob ajustat,
    baixes_detectades, probs_initial i probs_adjusted.
    """
    global games_full

    out = dict(result)
    out["probs_initial"] = {"over_25_prob": result.get("over_25_prob", 0)}
    out["baixes_detectades"] = []

    if games_full is None:
        return out

    key_map = games_full.get("key_players_per_club") or {}
    id_home = result.get("id_local")
    id_away = result.get("id_away")
    key_home = key_map.get(int(id_home), []) if id_home is not None else []
    key_away = key_map.get(int(id_away), []) if id_away is not None else []

    home_missing: list[tuple[str, str]] = []
    for inj in live_data.get("home_injuries") or []:
        name = inj.get("name") or ""
        match = _is_key_player_missing(name, key_home)
        if match:
            home_missing.append((match, (inj.get("reason") or "Lesió")))
    away_missing: list[tuple[str, str]] = []
    for inj in live_data.get("away_injuries") or []:
        name = inj.get("name") or ""
        match = _is_key_player_missing(name, key_away)
        if match:
            away_missing.append((match, (inj.get("reason") or "Lesió")))

    home_available = (live_data.get("home_lineup") or live_data.get("home_squad") or [])
    away_available = (live_data.get("away_lineup") or live_data.get("away_squad") or [])
    if home_available and not home_missing:
        for kp in _key_players_absent_from_squad(key_home, home_available):
            home_missing.append((kp, "No a l'alineació"))
    if away_available and not away_missing:
        for kp in _key_players_absent_from_squad(key_away, away_available):
            away_missing.append((kp, "No a l'alineació"))

    for nom, reason in home_missing:
        out["baixes_detectades"].append({"equip": "local", "nom": nom, "reason": reason})
    for nom, reason in away_missing:
        out["baixes_detectades"].append({"equip": "visitant", "nom": nom, "reason": reason})

    n_home = min(MAX_KEY_PLAYERS_PENALIZED, len(home_missing))
    n_away = min(MAX_KEY_PLAYERS_PENALIZED, len(away_missing))
    if n_home == 0 and n_away == 0:
        out["probs_adjusted"] = out["probs_initial"]
        return out

    # Penalització fixa 15% si hi ha qualsevol baixa (Top 3 golejadors/assistent o jugadors clau)
    # Així el canvi es veu clar al velocímetre
    PENALTY_FIXED_OVER25 = 0.15
    penalty = PENALTY_FIXED_OVER25
    p_over = float(result.get("over_25_prob", 50))
    p_over_adj = max(5, min(95, p_over * (1 - penalty)))
    out["over_25_prob"] = p_over_adj
    out["probs_adjusted"] = {"over_25_prob": p_over_adj}

    return out


def llistat_equips(clubs_df_ref: pd.DataFrame) -> list[str]:
    """Retorna la llista de noms d’equips únics ordenats (per selectors a la web)."""
    return sorted(clubs_df_ref["name"].drop_duplicates().astype(str).tolist())


def main() -> tuple[Any, pd.DataFrame, list[str]]:
    """
    Pipeline complet: carrega dades (incl. players per Squad Value), preprocessa, entrena.
    El model només s’entrena una vegada; l’app ha de fer servir st.cache_resource(main) o equivalent.
    """
    global model, feature_cols, clubs_df, games_full

    print("Carregant dades (clubs, partits, jugadors)...")
    clubs, club_games, games, players = carregar_dades()
    clubs_df = clubs

    print("Preprocessant (merge, rolling, Squad Value des de players + ESPN + competition_type)...")
    df, feature_cols_list, cg_chrono, clubs_processed, meta = fusionar_i_rolling(clubs, club_games, games, players)

    # Aplicar llista de features podades (generada per prune_features.py) si existeix
    if os.path.isfile(PATH_PRUNED_FEATURES):
        try:
            with open(PATH_PRUNED_FEATURES, encoding="utf-8") as f:
                pruned_set = {line.strip() for line in f if line.strip()}
            before = len(feature_cols_list)
            feature_cols_list = [c for c in feature_cols_list if c in pruned_set]
            if len(feature_cols_list) < before:
                print(f"  -> Features podades: {before} -> {len(feature_cols_list)} (fitxer {PATH_PRUNED_FEATURES})")
        except Exception as e:
            print(f"  -> No s'ha pogut llegir {PATH_PRUNED_FEATURES}: {e}")

    # Mitjanes de lliga (per H2H on-demand al predictor quan no hi ha historial)
    total_goals = df["home_club_goals"].astype(int) + df["away_club_goals"].astype(int)
    league_avg_goals = float(total_goals.mean())
    league_over25_rate = float((total_goals >= 3).mean())

    for col in feature_cols_list:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Últim estat per club: darrer partit a casa i darrer a fora (per predictor Home/Away)
    home_cols = [c for c in cg_chrono.columns if ("roll_" in c and "_home_" in c) or (c.startswith("home_advantage"))]
    away_cols = [c for c in cg_chrono.columns if ("roll_" in c and "_away_" in c) or (c.startswith("away_advantage"))]
    cg_chrono_sorted = cg_chrono.sort_values(["club_id", "date"])
    last_home = cg_chrono_sorted[cg_chrono_sorted["roll_own_goals_home_short"].notna()].groupby("club_id").last()[home_cols].reset_index() if "roll_own_goals_home_short" in cg_chrono.columns else pd.DataFrame(columns=["club_id"] + home_cols)
    last_away = cg_chrono_sorted[cg_chrono_sorted["roll_own_goals_away_short"].notna()].groupby("club_id").last()[away_cols].reset_index() if "roll_own_goals_away_short" in cg_chrono.columns else pd.DataFrame(columns=["club_id"] + away_cols)
    last_one = cg_chrono_sorted.groupby("club_id").last().reset_index()[["club_id", "club_market_value", "new_manager_effect"]]
    if not last_home.empty and not last_away.empty:
        last_stats = last_home.merge(last_away, on="club_id", how="outer").merge(last_one, on="club_id", how="left")
    elif not last_home.empty:
        last_stats = last_home.merge(last_one, on="club_id", how="left")
        for c in away_cols:
            last_stats[c] = np.nan
    elif not last_away.empty:
        last_stats = last_away.merge(last_one, on="club_id", how="left")
        for c in home_cols:
            last_stats[c] = np.nan
    else:
        last_stats = last_one.copy()
        last_stats["roll_own_goals_short"] = np.nan
        last_stats["roll_opponent_goals_short"] = np.nan
        last_stats["roll_punts_short"] = np.nan
        last_stats["roll_own_goals_long"] = last_stats["roll_opponent_goals_long"] = last_stats["roll_punts_long"] = np.nan
    def _mean_short(col_home: str, col_away: str, fallback: float) -> float:
        a = last_stats.get(col_home)
        b = last_stats.get(col_away)
        if a is not None and b is not None:
            return float(pd.concat([a.dropna(), b.dropna()]).mean())
        if a is not None:
            return float(a.mean())
        if b is not None:
            return float(b.mean())
        return fallback
    mean_roll_gf = _mean_short("roll_own_goals_home_short", "roll_own_goals_away_short", 1.0)
    mean_roll_ga = _mean_short("roll_opponent_goals_home_short", "roll_opponent_goals_away_short", 1.0)
    mean_roll_pts = _mean_short("roll_punts_home_short", "roll_punts_away_short", 1.0)
    mean_mv = last_stats["club_market_value"].mean()
    mean_new_manager = float(last_stats["new_manager_effect"].mean())

    # Jugadors clau per club (top per valor de mercat) per a l’ajust híbrid amb API (baixes)
    players_sorted = players.dropna(subset=["current_club_id", "market_value_in_eur"]).copy()
    players_sorted["market_value_in_eur"] = pd.to_numeric(players_sorted["market_value_in_eur"], errors="coerce").fillna(0)
    key_players_per_club: dict[int, list[str]] = {}
    for cid in players_sorted["current_club_id"].unique():
        sub = players_sorted[players_sorted["current_club_id"] == cid].nlargest(10, "market_value_in_eur")
        names = sub["name"].dropna().astype(str).str.strip().tolist()
        key_players_per_club[int(cid)] = names

    # Última data de partit per club (per al predictor: dies de descans)
    last_match_date_per_club = cg_chrono.groupby("club_id")["date"].last()
    mean_days_rest = float(df["home_days_rest"].mean()) if "home_days_rest" in df.columns else 7.0

    # Agregats ESPN per club (si s'han integrat)
    espn_cols = [c for c in df.columns if c.startswith("home_espn_")]
    espn_per_club = None
    espn_global_means: dict[str, float] = {}
    if espn_cols:
        # Derivar map per club_id des de per_club (cg_chrono ja els conté via fusionar_i_rolling)
        # Ens quedem amb últim registre per club amb les micro-dades ESPN
        cg_sorted_espn = cg_chrono.sort_values(["club_id", "date"]).copy()
        keep_cols = ["club_id"] + [c for c in cg_sorted_espn.columns if c.startswith("espn_")]
        cg_last_espn = cg_sorted_espn[keep_cols].drop_duplicates("club_id", keep="last").set_index("club_id")
        espn_per_club = cg_last_espn
        for c in cg_last_espn.columns:
            key = c.replace("espn_", "")
            espn_global_means[key] = float(cg_last_espn[c].mean())

    # Multi-league 2026 per club (Azure o CSV local, per al predictor)
    ml_per_club = None
    ml_global_means: dict[str, float] = {"ml_avg_corners": 5.0, "ml_avg_cards": 2.0, "ml_market_expectation": 0.5}
    ml_df = _load_summary_df("multi_league_2026_summary", PATH_MULTI_LEAGUE_2026)
    if ml_df is not None and not ml_df.empty:
        try:
            if "club_id" in ml_df.columns and ml_df["club_id"].notna().any():
                ml_df = ml_df.drop_duplicates("club_id").set_index("club_id")
                ml_df["ml_avg_corners"] = pd.to_numeric(ml_df.get("avg_corners", 0), errors="coerce").fillna(5.0)
                ml_df["ml_avg_cards"] = (
                    pd.to_numeric(ml_df.get("avg_yellows", 0), errors="coerce").fillna(0)
                    + 2.0 * pd.to_numeric(ml_df.get("avg_reds", 0), errors="coerce").fillna(0)
                )
                ml_df["ml_market_expectation"] = pd.to_numeric(ml_df.get("market_expectation", 0.5), errors="coerce").fillna(0.5)
                ml_per_club = ml_df[["ml_avg_corners", "ml_avg_cards", "ml_market_expectation"]]
                ml_global_means = {c: float(ml_per_club[c].mean()) for c in ml_per_club.columns}
        except Exception:
            pass

    # StatsBomb (Azure o CSV local) per al predictor
    sb_per_club: dict[int, float] = {}
    sb_tactical_danger_per_club: dict[int, float] = {}
    sb_df = _load_summary_df("statsbomb_summary", PATH_STATSBOMB_SUMMARY)
    if sb_df is not None and not sb_df.empty:
        try:
            sb_df["avg_xg"] = pd.to_numeric(sb_df["avg_xg"], errors="coerce").fillna(0.0)
            if "avg_key_passes" in sb_df.columns:
                sb_df["avg_key_passes"] = pd.to_numeric(sb_df["avg_key_passes"], errors="coerce").fillna(0.0)
            else:
                sb_df["avg_key_passes"] = 0.0
            sb_df["team_norm"] = sb_df["team_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            sb_df["tactical_danger_index"] = (sb_df["avg_xg"] * sb_df["avg_key_passes"]) / 100.0
            sb_means = sb_df.groupby("team_norm")["avg_xg"].mean()
            sb_td_means = sb_df.groupby("team_norm")["tactical_danger_index"].mean()
            for _, row in clubs_processed.iterrows():
                cid, name = row.get("club_id"), row.get("name")
                if pd.isna(cid) or pd.isna(name):
                    continue
                norm = " ".join(str(name).lower().strip().split())
                if norm in sb_means.index:
                    sb_per_club[int(cid)] = float(sb_means[norm])
                if norm in sb_td_means.index:
                    sb_tactical_danger_per_club[int(cid)] = float(sb_td_means[norm])
        except Exception:
            sb_per_club = {}
            sb_tactical_danger_per_club = {}

    games_full = {
        "games_df": games,
        "league_avg_goals": league_avg_goals,
        "league_over25_rate": league_over25_rate,
        "cg_chrono": cg_chrono,
        "clubs": clubs_processed,
        "last_stats_per_club": last_stats,
        "last_match_date_per_club": last_match_date_per_club,
        "mean_days_rest": mean_days_rest,
        "mean_roll_gf": mean_roll_gf,
        "mean_roll_ga": mean_roll_ga,
        "mean_roll_pts": mean_roll_pts,
        "mean_market_value": mean_mv,
        "mean_new_manager_effect": mean_new_manager,
        "key_players_per_club": key_players_per_club,
        "espn_per_club": espn_per_club,
        "espn_global_means": espn_global_means,
        "ml_per_club": ml_per_club,
        "ml_global_means": ml_global_means,
        "sb_per_club": sb_per_club,
        "sb_tactical_danger_per_club": sb_tactical_danger_per_club,
    }

    print("Entrenant model (XGBoost binary Over 2.5)...")
    global test_accuracy, test_confusion_matrix, feature_importance_df, baseline_accuracy

    # Entrenament base sense micro-dades ESPN (per comparar accuracy)
    base_features = [c for c in feature_cols_list if not (c.startswith("home_espn_") or c.startswith("away_espn_"))]
    if base_features:
        print("  -> Entrenant model base (sense ESPN) per calcular baseline_accuracy...")
        _, base_acc, _, _ = entrenar_model(df, base_features)
        baseline_accuracy = base_acc
        print(f"Baseline accuracy (sense ESPN): {base_acc:.4f}")

    model, acc, imp_df, conf_mat = entrenar_model(df, feature_cols_list)
    feature_cols = feature_cols_list
    test_accuracy = acc
    test_confusion_matrix = conf_mat
    feature_importance_df = imp_df
    if os.path.isdir(CSV_DIR):
        try:
            imp_df.to_csv(PATH_FEATURE_IMPORTANCE, index=False)
            print(f"  -> Importància guardada a {PATH_FEATURE_IMPORTANCE}")
        except Exception as e:
            print(f"  -> No s'ha pogut desar importància: {e}")
    print(f"\nAccuracy (test, amb ESPN): {acc:.4f}")

    return model, df, feature_cols_list, test_accuracy, test_confusion_matrix, feature_importance_df, baseline_accuracy


if __name__ == "__main__":
    main()
