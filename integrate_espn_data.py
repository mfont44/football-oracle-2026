from __future__ import annotations

"""
Preprocessament del dataset massiu ESPN Soccer Data 2024-2025
per generar un resum compacte per equip:

- Mitjana de possession, shotsOnTarget, fouls i corners per partit.
- Offensive Index = (shotsOnTarget / shotsTotal) * possession.
- Defensive Index = (saves / shotsOnTargetAgainst).

Dissenyat per treballar amb fitxers grans sense carregar-ho tot en memòria
utilitzant chunksize i dtypes optimitzats.

Output principal:
- csvfiles/espn_2025_summary.csv
"""

import os
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csvfiles")

# Estructura real del dump ESPN dins de csvfiles/
BASE_DATA_DIR = os.path.join(CSV_DIR, "base_data")
PLAYER_STATS_DIR = os.path.join(CSV_DIR, "playerStats_data")

# Fitxers base massius (per equip/lliga)
PATH_TEAM_STATS = os.path.join(BASE_DATA_DIR, "teamStats.csv")
PATH_FIXTURES = os.path.join(BASE_DATA_DIR, "fixtures.csv")

OUTPUT_SUMMARY = os.path.join(CSV_DIR, "espn_2025_summary.csv")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _normalize_name(name: Any) -> str:
    return " ".join(str(name).lower().strip().split())


def _iter_team_stats_chunks(chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    """
    Llegeix teamStats.csv en chunks amb dtypes compactes.
    No assumeix exactament els noms de columna; intenta adaptar-se.
    """
    if not os.path.isfile(PATH_TEAM_STATS):
        raise FileNotFoundError(f"No s'ha trobat teamStats.csv a {PATH_TEAM_STATS}")

    # Dtypes conservadors (pots ajustar-los si coneixes l'estructura exacta)
    dtypes = {
        "teamId": "Int32",
        "team_id": "Int32",
        "matchId": "Int32",
        "match_id": "Int32",
        "season": "Int32",
        "season_id": "Int32",
        "shotsOnTarget": "float32",
        "shotsOnTargetAgainst": "float32",
        "shotsTotal": "float32",
        "shotsTotalAgainst": "float32",
        "fouls": "float32",
        "corners": "float32",
        "saves": "float32",
        "possession": "float32",
    }

    usecols = list(dtypes.keys())

    for chunk in pd.read_csv(
        PATH_TEAM_STATS,
        chunksize=chunksize,
        dtype=dtypes,
        usecols=lambda c: c in usecols or c.lower() in [u.lower() for u in usecols],
        low_memory=False,
    ):
        yield chunk


def _build_team_aggregates() -> pd.DataFrame:
    """
    Construeix un DataFrame agregat per equip amb:
    - avg_possession
    - avg_shots_on_target
    - avg_shots_total
    - avg_fouls
    - avg_corners
    - offensive_index
    - defensive_index
    """
    agg_sums: Dict[str, pd.Series] = {}
    agg_counts: Dict[str, pd.Series] = {}

    # Noms possibles de columnes
    team_id_cols = ["teamId", "team_id"]

    for chunk in _iter_team_stats_chunks():
        cols = {c.lower(): c for c in chunk.columns}

        # Determinar columna de team_id
        team_col = None
        for cand in team_id_cols:
            if cand in chunk.columns:
                team_col = cand
                break
            if cand.lower() in cols:
                team_col = cols[cand.lower()]
                break
        if team_col is None:
            raise ValueError("No s'ha pogut identificar la columna d'equip (teamId / team_id) a teamStats.csv")

        # Filtrem només temporada 2024-2025 si tenim informació de temporada
        if "season" in chunk.columns:
            season_mask = chunk["season"].astype(str).str.contains("2024")
            if season_mask.any():
                chunk = chunk[season_mask]
        elif "season_id" in chunk.columns:
            season_mask = chunk["season_id"].astype(str).str.contains("2024")
            if season_mask.any():
                chunk = chunk[season_mask]

        if chunk.empty:
            continue

        # Assegurar float32
        for col in [
            "possession",
            "shotsOnTarget",
            "shotsOnTargetAgainst",
            "shotsTotal",
            "shotsTotalAgainst",
            "fouls",
            "corners",
            "saves",
        ]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

        # Agreguem només les columnes que realment existeixen al chunk
        metric_cols = [
            col
            for col in [
                "possession",
                "shotsOnTarget",
                "shotsOnTargetAgainst",
                "shotsTotal",
                "shotsTotalAgainst",
                "fouls",
                "corners",
                "saves",
            ]
            if col in chunk.columns
        ]
        if not metric_cols:
            continue

        group = chunk.groupby(team_col, as_index=True)[metric_cols].sum()
        counts = chunk.groupby(team_col, as_index=True).size().rename("n_matches")

        if not agg_sums:
            agg_sums = {c: group[c].copy() for c in group.columns}
            agg_counts["n_matches"] = counts.copy()
        else:
            for c in group.columns:
                if c in agg_sums:
                    agg_sums[c] = agg_sums[c].add(group[c], fill_value=0)
                else:
                    agg_sums[c] = group[c]
            agg_counts["n_matches"] = agg_counts["n_matches"].add(counts, fill_value=0)

    if not agg_sums:
        raise RuntimeError("No s'ha pogut construir cap agregat des de teamStats.csv (dataset buit?).")

    df = pd.DataFrame(agg_sums)
    df["n_matches"] = agg_counts["n_matches"].astype("int32")

    # Assegurar columna team_id robustament a partir de l'índex del groupby
    df = df.reset_index()
    if "team_id" in df.columns:
        pass
    elif "teamId" in df.columns:
        df = df.rename(columns={"teamId": "team_id"})
    else:
        # Si el nom de la columna ve de l'índex del groupby (p. ex. "clubId")
        if df.index.name and df.index.name in df.columns:
            df = df.rename(columns={df.index.name: "team_id"})
        else:
            # Agafar la primera columna que no sigui mètrica com a identificador
            candidate_cols = [
                c
                for c in df.columns
                if c
                not in {
                    "possession",
                    "shotsOnTarget",
                    "shotsOnTargetAgainst",
                    "shotsTotal",
                    "shotsTotalAgainst",
                    "fouls",
                    "corners",
                    "saves",
                    "n_matches",
                }
            ]
            if candidate_cols:
                df = df.rename(columns={candidate_cols[0]: "team_id"})
            else:
                df["team_id"] = np.arange(len(df), dtype="int32")

    # Mitjanes per partit
    with np.errstate(divide="ignore", invalid="ignore"):
        if "possession" in df.columns:
            df["avg_possession"] = (df["possession"] / df["n_matches"]).astype("float32")
        else:
            df["avg_possession"] = np.nan

        if "shotsOnTarget" in df.columns:
            df["avg_shots_on_target"] = (df["shotsOnTarget"] / df["n_matches"]).astype("float32")
        else:
            df["avg_shots_on_target"] = np.nan

        if "shotsTotal" in df.columns:
            df["avg_shots_total"] = (df["shotsTotal"] / df["n_matches"]).astype("float32")
        else:
            df["avg_shots_total"] = np.nan

        if "fouls" in df.columns:
            df["avg_fouls"] = (df["fouls"] / df["n_matches"]).astype("float32")
        else:
            df["avg_fouls"] = np.nan

        if "corners" in df.columns:
            df["avg_corners"] = (df["corners"] / df["n_matches"]).astype("float32")
        else:
            df["avg_corners"] = np.nan

        # Offensive Index: (shotsOnTarget / shotsTotal) * possession
        shots_on = df.get("shotsOnTarget", pd.Series(dtype="float32"))
        shots_total = df.get("shotsTotal", pd.Series(dtype="float32"))
        possession_sum = df.get("possession", pd.Series(dtype="float32"))

        ratio_shots = shots_on / shots_total.replace(0, np.nan)
        avg_poss = possession_sum / df["n_matches"].replace(0, np.nan)
        off_index = ratio_shots * avg_poss
        df["offensive_index"] = off_index.replace([np.inf, -np.inf], np.nan).astype("float32")

        # Defensive Index: (saves / shotsOnTargetAgainst)
        saves = df.get("saves", pd.Series(dtype="float32"))
        shots_against = df.get("shotsOnTargetAgainst", pd.Series(dtype="float32"))
        def_index = saves / shots_against.replace(0, np.nan)
        df["defensive_index"] = def_index.replace([np.inf, -np.inf], np.nan).astype("float32")

    if "team_id" in df.columns:
        df = df.drop_duplicates(subset=["team_id"])

    return df


def _add_micro_form_ewm(summary_df: pd.DataFrame, span_matches: int = 3) -> pd.DataFrame:
    """
    Calcula una "Forma de Micro-estats" per cada equip a partir de teamStats.csv:
    - EWM (span=3) de possessió i tirs a porteria ordenats per updateTime.
    - Ens quedem amb l'últim valor (partit més recent) per equip.
    """
    if not os.path.isfile(PATH_TEAM_STATS):
        return summary_df

    # Llegim només les columnes necessàries; aquest CSV és relativament manejable
    dtypes = {
        "teamId": "Int32",
        "possessionPct": "float32",
        "shotsOnTarget": "float32",
        "totalShots": "float32",
    }
    try:
        ts = pd.read_csv(
            PATH_TEAM_STATS,
            usecols=["teamId", "possessionPct", "shotsOnTarget", "totalShots", "updateTime"],
            dtype=dtypes,
            low_memory=False,
        )
    except Exception:
        return summary_df

    if ts.empty:
        return summary_df

    ts["updateTime"] = pd.to_datetime(ts["updateTime"], errors="coerce")
    ts = ts.dropna(subset=["teamId", "updateTime"])
    ts = ts.sort_values(["teamId", "updateTime"])

    def _ewm_last(g: pd.DataFrame) -> pd.Series:
        poss = g["possessionPct"].ewm(span=span_matches, adjust=False).mean()
        shots = g["shotsOnTarget"].ewm(span=span_matches, adjust=False).mean()
        total_shots = g["totalShots"].ewm(span=span_matches, adjust=False).mean()
        last = g.iloc[-1].copy()
        last["ewm_possessionPct"] = poss.iloc[-1]
        last["ewm_shotsOnTarget"] = shots.iloc[-1]
        last["ewm_totalShots"] = total_shots.iloc[-1]
        return last[["ewm_possessionPct", "ewm_shotsOnTarget", "ewm_totalShots"]]

    micro = ts.groupby("teamId", as_index=True).apply(_ewm_last)
    micro = micro.rename(
        columns={
            "ewm_possessionPct": "micro_possession_ewm3",
            "ewm_shotsOnTarget": "micro_shotsOnTarget_ewm3",
            "ewm_totalShots": "micro_totalShots_ewm3",
        }
    )

    summary_df = summary_df.copy()
    summary_df = summary_df.set_index("team_id")
    # Alignem per teamId -> team_id
    for col in micro.columns:
        summary_df[col] = micro[col]
    summary_df = summary_df.reset_index()
    return summary_df


def _attach_team_names(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta afegir el nom de l'equip (team_name) a partir dels fixtures
    per facilitar el mapping posterior amb clubs.csv.
    """
    if not os.path.isfile(PATH_FIXTURES):
        # No tenim fixtures -> només retornem l'ID
        summary_df["team_name"] = summary_df["team_id"].astype(str)
        return summary_df

    # Lectors en chunks per no carregar tot el fixtures.csv
    name_map: Dict[Any, str] = {}
    dtypes = {
        "homeTeamId": "Int32",
        "awayTeamId": "Int32",
        "homeTeamName": "string",
        "awayTeamName": "string",
    }

    for chunk in pd.read_csv(
        PATH_FIXTURES,
        chunksize=200_000,
        dtype=dtypes,
        low_memory=False,
    ):
        for side_id, side_name in [
            ("homeTeamId", "homeTeamName"),
            ("awayTeamId", "awayTeamName"),
        ]:
            if side_id not in chunk.columns or side_name not in chunk.columns:
                continue
            sub = chunk[[side_id, side_name]].dropna().drop_duplicates()
            sub = sub.astype({side_id: "Int32"})
            for _id, _name in zip(sub[side_id], sub[side_name]):
                if pd.isna(_id) or pd.isna(_name):
                    continue
                if _id not in name_map:
                    name_map[int(_id)] = str(_name)

    summary_df = summary_df.copy()
    summary_df["team_name"] = summary_df["team_id"].map(name_map).fillna(summary_df["team_id"].astype(str))
    return summary_df


def _touch_player_stats() -> None:
    """
    Llegeix playerStats.csv en chunks per confirmar que el fitxer és accessible
    i validar dtypes sense carregar-ho tot en memòria.
    (Reservat per futures millores amb micro-dades de jugadors).
    """
    if not os.path.isdir(PLAYER_STATS_DIR):
        return

    # Agafem alguns fitxers playerStats_* per validar que el directori existeix
    try:
        files = [f for f in os.listdir(PLAYER_STATS_DIR) if f.lower().startswith("playerstats_") and f.lower().endswith(".csv")]
    except OSError:
        return

    if not files:
        return

    sample_path = os.path.join(PLAYER_STATS_DIR, files[0])
    dtypes = {
        "playerId": "Int32",
        "teamId": "Int32",
        "minutes": "Int32",
        "goals": "Int16",
        "assists": "Int16",
    }
    # Només llegim uns quants chunks d'un fitxer per validar estructura
    for i, _ in enumerate(
        pd.read_csv(
            sample_path,
            chunksize=200_000,
            dtype=dtypes,
            low_memory=False,
        )
    ):
        if i >= 2:
            break


def main() -> pd.DataFrame:
    """
    Pipeline principal:
    1) Agrega teamStats.csv per equip (mitjanes i índexos).
    2) Adjunta noms d'equip a partir de fixtures.csv (si existeix).
    3) Escriu csvfiles/espn_2025_summary.csv.
    """
    print("Construint agregats ESPN 2024-2025 per equip (teamStats)...")
    summary = _build_team_aggregates()
    summary = _add_micro_form_ewm(summary)
    summary = _attach_team_names(summary)

    # Validació ràpida
    cols_order = [
        "team_id",
        "team_name",
        "n_matches",
        "avg_possession",
        "avg_shots_on_target",
        "avg_shots_total",
        "avg_fouls",
        "avg_corners",
        "offensive_index",
        "defensive_index",
    ]
    for c in cols_order:
        if c not in summary.columns:
            summary[c] = np.nan

    summary = summary[cols_order]
    summary = summary.drop_duplicates(subset=["team_id"])

    _ensure_dir(OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"Fitxer resumit creat: {OUTPUT_SUMMARY} (files: {len(summary)})")

    # Escanejar mínimament playerStats.csv per assegurar accessibilitat
    _touch_player_stats()

    return summary


if __name__ == "__main__":
    main()

