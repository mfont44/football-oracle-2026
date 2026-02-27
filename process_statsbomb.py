# -*- coding: utf-8 -*-
"""
Processa dades StatsBomb a csvfiles/data/ sense carregar tot en memòria (12.9GB JSON).
- Obre fitxers de data/events/ un per un (iterativament).
- Per a cada partit: suma statsbomb_xg dels esdeveniments Shot; compta through_balls i deep_completions per equip.
- Guarda el resum a statsbomb_summary.csv: team_name, season_id, avg_xg, avg_key_passes.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

# Rutes respecte a la carpeta del projecte
DATA_DIR = os.path.join("csvfiles", "data")
EVENTS_DIR = os.path.join(DATA_DIR, "events")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")
OUTPUT_PATH = os.path.join("csvfiles", "statsbomb_summary.csv")

# Deep completion: passada que acaba a zona d'atac (StatsBomb: longitud >= 80)
DEEP_COMPLETION_X_THRESHOLD = 80.0


def _match_id_from_filename(name: str) -> int | None:
    if not name.endswith(".json"):
        return None
    try:
        return int(name[:-5])
    except ValueError:
        return None


def build_match_to_season() -> dict[int, tuple[int, int]]:
    """Recorre data/matches/<comp_id>/<season_id>.json; retorna match_id -> (season_id, season_year)."""
    match_to_season: dict[int, tuple[int, int]] = {}
    if not os.path.isdir(MATCHES_DIR):
        return match_to_season
    for comp_name in os.listdir(MATCHES_DIR):
        comp_path = os.path.join(MATCHES_DIR, comp_name)
        if not os.path.isdir(comp_path):
            continue
        for season_name in os.listdir(comp_path):
            path = os.path.join(comp_path, season_name)
            if not path.endswith(".json"):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            try:
                season_id = int(season_name[:-5])
            except ValueError:
                continue
            for m in data:
                if not isinstance(m, dict):
                    continue
                mid = m.get("match_id")
                if mid is None:
                    continue
                year = season_id  # fallback
                md = m.get("match_date")
                if isinstance(md, str) and len(md) >= 4:
                    try:
                        year = int(md[:4])
                    except ValueError:
                        pass
                match_to_season[int(mid)] = (season_id, year)
    return match_to_season


def process_one_match_events(
    filepath: str, match_id: int, season_id: int, season_year: int
) -> list[tuple[str, float, int, int]]:
    """
    Processa un fitxer d'esdeveniments. Retorna llista de (team_name, total_xg, through_balls, deep_completions) per equip.
    Cada partit té 2 equips; retornem 2 tuples.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            events = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(events, list):
        return []

    # Per equip: (total_xg, through_balls, deep_completions)
    agg: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0, 0])

    for ev in events:
        if not isinstance(ev, dict):
            continue
        team = ev.get("team")
        if not isinstance(team, dict):
            continue
        team_name = team.get("name")
        if not team_name or not isinstance(team_name, str):
            continue

        typ = ev.get("type")
        type_name = typ.get("name") if isinstance(typ, dict) else None

        if type_name == "Shot":
            shot = ev.get("shot")
            if isinstance(shot, dict) and "statsbomb_xg" in shot:
                xg = float(shot["statsbomb_xg"])
                agg[team_name][0] += xg

        elif type_name == "Pass":
            pass_obj = ev.get("pass")
            if isinstance(pass_obj, dict):
                if pass_obj.get("through_ball") is True:
                    agg[team_name][1] += 1
                end_loc = pass_obj.get("end_location")
                if isinstance(end_loc, list) and len(end_loc) >= 1:
                    try:
                        x = float(end_loc[0])
                        if x >= DEEP_COMPLETION_X_THRESHOLD:
                            agg[team_name][2] += 1
                    except (TypeError, ValueError):
                        pass

    return [(t, a[0], a[1], a[2]) for t, a in agg.items()]


def main() -> None:
    match_to_season = build_match_to_season()
    print(f"Map match_id -> (season_id, year): {len(match_to_season)} partits.")

    # Agregat (team_name, season_id, season_year) -> (total_xg, total_through_balls, total_deep_completions, n_matches)
    summary: dict[tuple[str, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    if not os.path.isdir(EVENTS_DIR):
        print(f"No s'ha trobat {EVENTS_DIR}.")
        return

    event_files = [f for f in os.listdir(EVENTS_DIR) if f.endswith(".json")]
    processed = 0
    skipped_no_season = 0

    for fname in event_files:
        match_id = _match_id_from_filename(fname)
        if match_id is None:
            continue
        season_info = match_to_season.get(match_id)
        if season_info is None:
            skipped_no_season += 1
            continue
        season_id, season_year = season_info
        filepath = os.path.join(EVENTS_DIR, fname)
        rows = process_one_match_events(filepath, match_id, season_id, season_year)
        for team_name, total_xg, through_balls, deep_completions in rows:
            key = (team_name, season_id, season_year)
            summary[key][0] += total_xg
            summary[key][1] += through_balls
            summary[key][2] += deep_completions
            summary[key][3] += 1
        processed += 1
        if processed % 500 == 0:
            print(f"  Processats {processed} partits...")

    print(f"Partits processats: {processed}. Sense temporada al map: {skipped_no_season}.")

    # Construir CSV: team_name, season_id, season_year, avg_xg, avg_key_passes (season_year per encreuar amb model)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.write("team_name,season_id,season_year,avg_xg,avg_key_passes\n")
        for (team_name, season_id, season_year), (total_xg, tb, dc, n) in sorted(
            summary.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])
        ):
            n = max(1, n)
            avg_xg = total_xg / n
            avg_key_passes = (tb + dc) / n
            out.write(f"{team_name!s},{season_id},{season_year},{avg_xg:.6f},{avg_key_passes:.4f}\n")

    print(f"Resum guardat a {OUTPUT_PATH} ({len(summary)} files team-season).")


if __name__ == "__main__":
    main()
