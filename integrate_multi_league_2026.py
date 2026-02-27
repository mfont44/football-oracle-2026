# -*- coding: utf-8 -*-
"""
Integració dels datasets de les 5 grans lligues (LaLiga, Premier, Bundes, League1, SerieA)
per generar un resum per equip amb micro-stats i market_expectation.

Llegeix tots els CSV de forma recursiva dins de cada carpeta de lliga.

Columnes mapejades:
- Gols: FTHG (local), FTAG (visitant)
- Micro-stats: HS/AS (tirs totals), HST/AST (tirs a porta), HC/AC (córners)
- Disciplina: HY/AY (grogues), HR/AR (vermelles)
- Odds: Avg>2.5/Avg<2.5 → market_expectation; fallback AvgH, AvgD, AvgA si cal

Output: csvfiles/multi_league_2026_summary.csv
Mapping: fuzzy match amb clubs.csv (difflib) per obtenir club_id.
"""

from __future__ import annotations

import os
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csvfiles")
PATH_CLUBS = os.path.join(CSV_DIR, "clubs.csv")
LEAGUE_DIRS = [
    os.path.join(CSV_DIR, "LaLiga"),
    os.path.join(CSV_DIR, "Premier"),
    os.path.join(CSV_DIR, "Bundes"),
    os.path.join(CSV_DIR, "League1"),
    os.path.join(CSV_DIR, "SerieA"),
]
OUTPUT_PATH = os.path.join(CSV_DIR, "multi_league_2026_summary.csv")

# Columnes esperades (noms flexibles: pot ser HST o ShotsOnTarget etc.)
GOALS_HOME = "FTHG"
GOALS_AWAY = "FTAG"
TEAM_HOME = "HomeTeam"
TEAM_AWAY = "AwayTeam"
SHOTS_HOME = "HS"
SHOTS_AWAY = "AS"
SHOTS_TARGET_HOME = "HST"
SHOTS_TARGET_AWAY = "AST"
CORNERS_HOME = "HC"
CORNERS_AWAY = "AC"
YELLOW_HOME = "HY"
YELLOW_AWAY = "AY"
RED_HOME = "HR"
RED_AWAY = "AR"
ODDS_AVG_H = "AvgH"
ODDS_AVG_D = "AvgD"
ODDS_AVG_A = "AvgA"
ODDS_B365_H = "B365H"
ODDS_B365_D = "B365D"
ODDS_B365_A = "B365A"
ODDS_AVG_OVER25 = "Avg>2.5"
ODDS_AVG_UNDER25 = "Avg<2.5"


def _normalize(s: Any) -> str:
    return " ".join(str(s).lower().strip().split())


# Abreviatures de club que sovint varien (FC, CF, etc.)
_STRIP_TOKENS = re.compile(
    r"\b(fc|cf|sc|ssc|as|ac|og|cfc|sv|tsg)\b",
    re.IGNORECASE,
)


def _normalize_for_match(s: str) -> str:
    """Normalització per al fuzzy match: sense abreviatures com FC/CF que varien entre fonts."""
    s = _normalize(s)
    s = _STRIP_TOKENS.sub("", s).strip()
    return " ".join(s.split()) if s else s


def _find_csvs() -> List[str]:
    """Recull tots els fitxers CSV dins de les carpetes de lliga, de forma recursiva."""
    paths = []
    for league_dir in LEAGUE_DIRS:
        if not os.path.isdir(league_dir):
            continue
        for root, _dirs, files in os.walk(league_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    paths.append(os.path.join(root, f))
    return sorted(paths)


def _read_league_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols_lower = {c.strip(): c for c in df.columns}
    # Map expected columns (allow slight name variations)
    renames = {}
    for expected in [
        GOALS_HOME, GOALS_AWAY, TEAM_HOME, TEAM_AWAY,
        SHOTS_HOME, SHOTS_AWAY, SHOTS_TARGET_HOME, SHOTS_TARGET_AWAY,
        CORNERS_HOME, CORNERS_AWAY, YELLOW_HOME, YELLOW_AWAY, RED_HOME, RED_AWAY,
        ODDS_AVG_H, ODDS_AVG_D, ODDS_AVG_A,
        ODDS_B365_H, ODDS_B365_D, ODDS_B365_A,
        ODDS_AVG_OVER25, ODDS_AVG_UNDER25,
    ]:
        key = expected.lower().replace(" ", "")
        for k, v in cols_lower.items():
            if k.lower().replace(" ", "") == key:
                renames[v] = expected
                break
    df = df.rename(columns=renames)
    return df


def _aggregate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega estadístiques per equip (com a local i visitant)."""
    if df.empty:
        return pd.DataFrame()

    required = [TEAM_HOME, TEAM_AWAY, GOALS_HOME, GOALS_AWAY]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    # Normalitzar noms
    df = df.copy()
    df["_home"] = df[TEAM_HOME].astype(str).map(_normalize)
    df["_away"] = df[TEAM_AWAY].astype(str).map(_normalize)
    df[GOALS_HOME] = pd.to_numeric(df[GOALS_HOME], errors="coerce").fillna(0)
    df[GOALS_AWAY] = pd.to_numeric(df[GOALS_AWAY], errors="coerce").fillna(0)

    rows: List[Dict[str, Any]] = []
    seen_teams: Dict[str, Dict[str, Any]] = {}

    def add_match(team_norm: str, team_raw: str, is_home: bool, row: pd.Series):
        if team_norm not in seen_teams:
            seen_teams[team_norm] = {
                "team_name": team_raw,
                "n_matches": 0,
                "goals_for": 0.0,
                "goals_against": 0.0,
                "shots": 0.0,
                "shots_target": 0.0,
                "corners": 0.0,
                "yellows": 0.0,
                "reds": 0.0,
                "market_expectation_sum": 0.0,
                "market_expectation_n": 0,
            }
        r = seen_teams[team_norm]
        r["n_matches"] += 1
        if is_home:
            r["goals_for"] += row[GOALS_HOME]
            r["goals_against"] += row[GOALS_AWAY]
            if SHOTS_HOME in df.columns:
                r["shots"] += pd.to_numeric(row.get(SHOTS_HOME, 0), errors="coerce") or 0
            if SHOTS_AWAY in df.columns:
                pass  # away shots for opponent
            if SHOTS_TARGET_HOME in df.columns:
                r["shots_target"] += pd.to_numeric(row.get(SHOTS_TARGET_HOME, 0), errors="coerce") or 0
            if CORNERS_HOME in df.columns:
                r["corners"] += pd.to_numeric(row.get(CORNERS_HOME, 0), errors="coerce") or 0
            if YELLOW_HOME in df.columns:
                r["yellows"] += pd.to_numeric(row.get(YELLOW_HOME, 0), errors="coerce") or 0
            if RED_HOME in df.columns:
                r["reds"] += pd.to_numeric(row.get(RED_HOME, 0), errors="coerce") or 0
        else:
            r["goals_for"] += row[GOALS_AWAY]
            r["goals_against"] += row[GOALS_HOME]
            if SHOTS_AWAY in df.columns:
                r["shots"] += pd.to_numeric(row.get(SHOTS_AWAY, 0), errors="coerce") or 0
            if SHOTS_TARGET_AWAY in df.columns:
                r["shots_target"] += pd.to_numeric(row.get(SHOTS_TARGET_AWAY, 0), errors="coerce") or 0
            if CORNERS_AWAY in df.columns:
                r["corners"] += pd.to_numeric(row.get(CORNERS_AWAY, 0), errors="coerce") or 0
            if YELLOW_AWAY in df.columns:
                r["yellows"] += pd.to_numeric(row.get(YELLOW_AWAY, 0), errors="coerce") or 0
            if RED_AWAY in df.columns:
                r["reds"] += pd.to_numeric(row.get(RED_AWAY, 0), errors="coerce") or 0

        # Market expectation: implied P(Over 2.5) from Avg>2.5 / Avg<2.5
        if ODDS_AVG_OVER25 in df.columns and ODDS_AVG_UNDER25 in df.columns:
            o25 = pd.to_numeric(row.get(ODDS_AVG_OVER25), errors="coerce")
            u25 = pd.to_numeric(row.get(ODDS_AVG_UNDER25), errors="coerce")
            if pd.notna(o25) and pd.notna(u25) and o25 > 0 and u25 > 0:
                inv_o = 1.0 / float(o25)
                inv_u = 1.0 / float(u25)
                p_over = inv_o / (inv_o + inv_u)
                r["market_expectation_sum"] += p_over
                r["market_expectation_n"] += 1
        # Fallback: from AvgH, AvgD, AvgA (1X2) aproximem P(Over 2.5)
        elif all(c in df.columns for c in (ODDS_AVG_H, ODDS_AVG_D, ODDS_AVG_A)):
            ah = pd.to_numeric(row.get(ODDS_AVG_H), errors="coerce")
            ad = pd.to_numeric(row.get(ODDS_AVG_D), errors="coerce")
            aa = pd.to_numeric(row.get(ODDS_AVG_A), errors="coerce")
            if pd.notna(ah) and pd.notna(ad) and pd.notna(aa) and ah > 0 and ad > 0 and aa > 0:
                inv_h, inv_d, inv_a = 1.0 / float(ah), 1.0 / float(ad), 1.0 / float(aa)
                tot = inv_h + inv_d + inv_a
                p_draw = inv_d / tot
                # Heurística: E[gols] ~ 2.0 + 0.4*(1 - p_empat); P(Over 2.5) ~ sigmoide
                e_goals = 2.0 + 0.4 * (1.0 - p_draw)
                p_over = 1.0 / (1.0 + np.exp(-1.5 * (e_goals - 2.5)))
                r["market_expectation_sum"] += p_over
                r["market_expectation_n"] += 1

    for _, row in df.iterrows():
        add_match(row["_home"], row[TEAM_HOME], True, row)
        add_match(row["_away"], row[TEAM_AWAY], False, row)

    if not seen_teams:
        return pd.DataFrame()

    out = []
    for team_norm, r in seen_teams.items():
        n = r["n_matches"]
        if n == 0:
            continue
        row = {
            "team_name_norm": team_norm,
            "team_name": r["team_name"],
            "n_matches": n,
            "avg_goals_for": r["goals_for"] / n,
            "avg_goals_against": r["goals_against"] / n,
            "avg_shots": r["shots"] / n,
            "avg_shots_target": r["shots_target"] / n,
            "avg_corners": r["corners"] / n,
            "avg_yellows": r["yellows"] / n,
            "avg_reds": r["reds"] / n,
        }
        if r["market_expectation_n"] > 0:
            row["market_expectation"] = r["market_expectation_sum"] / r["market_expectation_n"]
        else:
            row["market_expectation"] = np.nan
        out.append(row)

    return pd.DataFrame(out)


def _best_match(name: str, ref_list: List[Tuple[str, str]]) -> Optional[Tuple[str, float]]:
    """
    Retorna el millor match (ref_norm, ratio) amb difflib.
    ref_list: [(name_norm, name_raw), ...]. Es prova amb name_norm i name_for_match (sense FC/CF).
    """
    name_norm = _normalize(name)
    name_for_match = _normalize_for_match(name)
    ref_norms = [r[0] for r in ref_list]
    candidates = difflib.get_close_matches(name_norm, ref_norms, n=5, cutoff=0.45)
    if not candidates:
        candidates = difflib.get_close_matches(name_for_match, ref_norms, n=5, cutoff=0.40)
    if not candidates:
        return None
    ref_by_norm = {r[0]: r for r in ref_list}
    best_ref = None
    best_ratio = 0.0
    for c in candidates:
        ref_norm, ref_raw = ref_by_norm.get(c, (c, c))
        ratio_norm = difflib.SequenceMatcher(None, name_norm, ref_norm).ratio()
        ratio_stripped = difflib.SequenceMatcher(None, name_for_match, _normalize_for_match(ref_raw)).ratio()
        ratio = max(ratio_norm, ratio_stripped)
        if ratio > best_ratio:
            best_ratio = ratio
            best_ref = ref_norm
    if best_ref is None or best_ratio < 0.45:
        return None
    return (best_ref, best_ratio)


def _map_to_clubs(summary: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy match team_name_norm / team_name a clubs.csv per obtenir club_id (difflib robust)."""
    if not os.path.isfile(PATH_CLUBS):
        summary["club_id"] = np.arange(len(summary), dtype="int32")
        return summary

    clubs = pd.read_csv(PATH_CLUBS, usecols=["club_id", "name"], dtype={"club_id": "int32", "name": "string"}, low_memory=False)
    clubs["name_norm"] = clubs["name"].astype(str).map(_normalize)
    # Llista única (norm, raw) per al fuzzy match; norm -> club_id per resoldre empat
    norm_to_cid: Dict[str, int] = {}
    ref_list: List[Tuple[str, str]] = []
    for _, r in clubs.drop_duplicates("name_norm", keep="first").iterrows():
        norm = r["name_norm"]
        raw = str(r["name"])
        norm_to_cid[norm] = int(r["club_id"])
        ref_list.append((norm, raw))

    club_ids = []
    for _, row in summary.iterrows():
        name_raw = str(row.get("team_name", ""))
        match_result = _best_match(name_raw, ref_list)
        if match_result:
            ref_norm, _ = match_result
            cid = norm_to_cid.get(ref_norm)
            club_ids.append(int(cid) if cid is not None else pd.NA)
        else:
            club_ids.append(pd.NA)

    summary = summary.copy()
    summary["club_id"] = club_ids
    return summary


def main() -> pd.DataFrame:
    paths = _find_csvs()
    if not paths:
        print("No s'han trobat CSV a cap de les carpetes de lliga (LaLiga, Premier, Bundes, League1, SerieA).")
        return pd.DataFrame()

    all_aggs: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = _read_league_csv(path)
            agg = _aggregate_team_stats(df)
            if not agg.empty:
                all_aggs.append(agg)
        except Exception as e:
            print(f"Error llegint {path}: {e}")

    if not all_aggs:
        print("Cap agregat generat.")
        return pd.DataFrame()

    # Combinar: si un equip surt a més d'una lliga, mitjana ponderada per n_matches
    combined = pd.concat(all_aggs, ignore_index=True)
    n_tot = combined.groupby("team_name_norm")["n_matches"].transform("sum")
    combined["_w"] = combined["n_matches"] / n_tot
    combined["_gf"] = combined["avg_goals_for"] * combined["n_matches"]
    combined["_ga"] = combined["avg_goals_against"] * combined["n_matches"]
    combined["_sh"] = combined["avg_shots"] * combined["n_matches"]
    combined["_sht"] = combined["avg_shots_target"] * combined["n_matches"]
    combined["_cr"] = combined["avg_corners"] * combined["n_matches"]
    combined["_yl"] = combined["avg_yellows"] * combined["n_matches"]
    combined["_rd"] = combined["avg_reds"] * combined["n_matches"]
    agg = combined.groupby("team_name_norm").agg(
        team_name=("team_name", "first"),
        n_matches=("n_matches", "sum"),
        avg_goals_for=("_gf", "sum"),
        avg_goals_against=("_ga", "sum"),
        avg_shots=("_sh", "sum"),
        avg_shots_target=("_sht", "sum"),
        avg_corners=("_cr", "sum"),
        avg_yellows=("_yl", "sum"),
        avg_reds=("_rd", "sum"),
        market_expectation=("market_expectation", "mean"),
    ).reset_index()
    agg["avg_goals_for"] = agg["avg_goals_for"] / agg["n_matches"]
    agg["avg_goals_against"] = agg["avg_goals_against"] / agg["n_matches"]
    agg["avg_shots"] = agg["avg_shots"] / agg["n_matches"]
    agg["avg_shots_target"] = agg["avg_shots_target"] / agg["n_matches"]
    agg["avg_corners"] = agg["avg_corners"] / agg["n_matches"]
    agg["avg_yellows"] = agg["avg_yellows"] / agg["n_matches"]
    agg["avg_reds"] = agg["avg_reds"] / agg["n_matches"]

    combined = _map_to_clubs(agg)
    combined = combined.dropna(subset=["club_id"])
    combined["club_id"] = combined["club_id"].astype("int32")
    combined = combined.drop_duplicates(subset=["club_id"], keep="first")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Resum guardat: {OUTPUT_PATH} ({len(combined)} equips europeus)")
    return combined


if __name__ == "__main__":
    main()
