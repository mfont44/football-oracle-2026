# -*- coding: utf-8 -*-
"""
Integració football-data.org (v4) per a dades en temps real: partits, alineacions, squad.
Autenticació: header X-Auth-Token. Mapping d'equips per nom (fuzzy match) per obtenir team_id.
Si l'API no dóna lesionats, es comparen jugadors disponibles (squad/lineup) amb players.csv per detectar absències.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

API_BASE = "https://api.football-data.org/v4"
REQUEST_TIMEOUT = 10
DEFAULT_API_KEY = "e9e39e8bfc8b48188963e3a0697eed09"


def _request(
    endpoint: str,
    params: Optional[dict[str, Any]] = None,
    api_key: str = DEFAULT_API_KEY,
) -> Optional[dict[str, Any]]:
    """
    Crida l'API football-data.org amb X-Auth-Token. Timeout i gestió d'errors.
    """
    if not api_key or api_key.strip() in (DEFAULT_API_KEY.strip(), "e9e39e8bfc8b48188963e3a0697eed09", ""):
        return None
    try:
        import json
        import urllib.error
        import urllib.parse
        import urllib.request

        url = f"{API_BASE}/{endpoint}"
        if params:
            qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
            url = f"{url}?{qs}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("X-Auth-Token", api_key.strip())

        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            logger.warning("football-data.org: token invàlid")
        elif e.code == 429:
            logger.warning("football-data.org: límit de peticions")
        else:
            logger.warning("football-data.org HTTP error: %s", e.code)
        return None
    except Exception as e:
        logger.warning("football-data.org request failed: %s", e)
        return None


def _normalize_name(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _fuzzy_match_team_name(candidate_name: str, target_name: str) -> float:
    """
    Retorna un score de similitud (0-1). Com més alt, millor coincidència.
    Basat en substring i paraules en comú.
    """
    c = _normalize_name(candidate_name)
    t = _normalize_name(target_name)
    if c == t:
        return 1.0
    if t in c or c in t:
        return 0.9
    c_words = set(c.split())
    t_words = set(t.split())
    common = c_words & t_words
    if not t_words:
        return 0.0
    return len(common) / len(t_words)


def get_matches(
    api_key: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Llista de partits. Per defecte avui (UTC).
    date_from / date_to: YYYY-MM-DD o None (avui).
    """
    params: dict[str, Any] = {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if date_from:
        params["dateFrom"] = date_from
    else:
        params["dateFrom"] = today
    if date_to:
        params["dateTo"] = date_to
    else:
        # per defecte 7 dies per trobar partits
        d = datetime.now(timezone.utc) + timedelta(days=7)
        params["dateTo"] = d.strftime("%Y-%m-%d")
    data = _request("matches", params, api_key)
    if not data or "matches" not in data:
        return []
    return data.get("matches") or []


def find_team_id_by_name(
    team_name: str,
    api_key: str,
    matches: Optional[list[dict[str, Any]]] = None,
) -> Optional[int]:
    """
    Busca el team_id de l'API per nom d'equip (fuzzy match).
    Recull equips des de la llista de partits (o fa una crida a matches) i retorna el millor match.
    """
    if not api_key or api_key.strip() in (DEFAULT_API_KEY.strip(), "e9e39e8bfc8b48188963e3a0697eed09", ""):
        return None
    if matches is None:
        matches = get_matches(api_key)
    target = _normalize_name(team_name)
    best_id: Optional[int] = None
    best_score = 0.0
    seen: set[int] = set()
    for m in matches:
        for side in ("homeTeam", "awayTeam"):
            team = m.get(side)
            if not isinstance(team, dict):
                continue
            tid = team.get("id")
            name = (team.get("name") or "").strip()
            if not name or tid is None:
                continue
            if tid in seen:
                continue
            seen.add(tid)
            score = _fuzzy_match_team_name(name, team_name)
            if score > best_score and score >= 0.3:
                best_score = score
                best_id = int(tid)
    return best_id


def get_match_between_teams(
    home_team_id: int,
    away_team_id: int,
    api_key: str,
    matches: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """Retorna el partit (objecte match) entre els dos equips si existeix a la llista."""
    if matches is None:
        matches = get_matches(api_key)
    for m in matches:
        h = (m.get("homeTeam") or {}).get("id")
        a = (m.get("awayTeam") or {}).get("id")
        if h is not None and a is not None and {int(h), int(a)} == {home_team_id, away_team_id}:
            return m
    return None


def get_match_by_id(match_id: int, api_key: str) -> Optional[dict[str, Any]]:
    """Detall d’un partit (incl. lineups si el pla ho permet)."""
    return _request(f"matches/{match_id}", None, api_key)


def get_team_squad(team_id: int, api_key: str) -> list[str]:
    """
    Retorna la llista de noms de jugadors del squad (GET /teams/{id}).
    Si el pla gratuït no ho permet, retorna llista buida.
    """
    data = _request(f"teams/{team_id}", None, api_key)
    if not data or "squad" not in data:
        return []
    names = []
    for p in data.get("squad") or []:
        if isinstance(p, dict):
            n = (p.get("name") or "").strip()
            if n:
                names.append(n)
    return names


def _lineup_plus_bench(match: dict[str, Any], side: str) -> list[str]:
    """Extrau noms de jugadors (lineup + bench) del partit per homeTeam o awayTeam."""
    team = match.get(side)
    if not isinstance(team, dict):
        return []
    names = []
    for key in ("lineup", "bench"):
        for item in team.get(key) or []:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item.get("name", "")).strip())
    return [n for n in names if n]


def get_live_data(
    team_name: str,
    api_key: str = DEFAULT_API_KEY,
    _season: Optional[int] = None,
) -> dict[str, Any]:
    """
    Dades en temps real per a un equip (football-data.org).
    Retorna: { "injuries": [], "lineup": None }.
    Les lesions no venen directament al pla gratuït; es poden inferir més endavant per comparació de squad.
    """
    out: dict[str, Any] = {"injuries": [], "lineup": None}
    matches = get_matches(api_key)
    team_id = find_team_id_by_name(team_name, api_key, matches)
    if team_id is None:
        return out
    out["squad_names"] = get_team_squad(team_id, api_key)
    return out


def get_live_data_for_match(
    home_team_name: str,
    away_team_name: str,
    api_key: str = DEFAULT_API_KEY,
    season: Optional[int] = None,
    date: Optional[str] = None,
) -> dict[str, Any]:
    """
    Dades en temps real per al partit (football-data.org v4).
    Prioritza Matches per estat, jornada i alineacions (lineup/bench).
    Si no hi ha lesionats directes, es retornen els noms del squad/lineup per comparar amb players.csv
    i detectar absències (jugadors clau que no surten a l’alineació).
    Retorna: home_injuries, away_injuries (inferits si cal), home_lineup, away_lineup, fixture_id, error, sync_log.
    """
    result: dict[str, Any] = {
        "home_injuries": [],
        "away_injuries": [],
        "home_lineup": None,
        "away_lineup": None,
        "home_squad": [],
        "away_squad": [],
        "fixture_id": None,
        "matchday": None,
        "status": None,
        "error": None,
        "sync_log": [],
    }
    if not api_key or api_key.strip() in (DEFAULT_API_KEY.strip(), "e9e39e8bfc8b48188963e3a0697eed09", ""):
        result["error"] = "API key no configurada"
        result["sync_log"].append("Sincronització: API key no configurada.")
        return result

    try:
        log: list[str] = []
        log.append(f"Intentant sincronitzar local: «{home_team_name}»")
        log.append(f"Intentant sincronitzar visitant: «{away_team_name}»")

        date_from = date
        date_to = None
        if date_from:
            from datetime import datetime as dt
            d = dt.strptime(date_from, "%Y-%m-%d").date()
            from datetime import timedelta
            date_to = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        matches = get_matches(api_key, date_from=date_from, date_to=date_to)
        if not matches:
            log.append("  → L’API no ha retornat partits (comprova dates o connexió).")
        id_home = find_team_id_by_name(home_team_name, api_key, matches)
        id_away = find_team_id_by_name(away_team_name, api_key, matches)

        if id_home is None:
            log.append(f"  → Local: cap equip de l’API coincideix amb «{home_team_name}». Revisa el nom.")
        else:
            log.append(f"  → Local: team_id={id_home} (match OK).")
        if id_away is None:
            log.append(f"  → Visitant: cap equip de l’API coincideix amb «{away_team_name}». Revisa el nom.")
        else:
            log.append(f"  → Visitant: team_id={id_away} (match OK).")

        result["sync_log"] = log

        if id_home is None or id_away is None:
            if id_home is not None:
                result["home_squad"] = get_team_squad(id_home, api_key)
            if id_away is not None:
                result["away_squad"] = get_team_squad(id_away, api_key)
            return result

        match = get_match_between_teams(id_home, id_away, api_key, matches)
        if match:
            result["fixture_id"] = match.get("id")
            result["matchday"] = match.get("matchday")
            result["status"] = match.get("status")
            home_line = _lineup_plus_bench(match, "homeTeam")
            away_line = _lineup_plus_bench(match, "awayTeam")
            if home_line or away_line:
                result["home_lineup"] = home_line
                result["away_lineup"] = away_line
            if not result["home_lineup"] and not result["away_lineup"]:
                result["home_squad"] = get_team_squad(id_home, api_key)
                result["away_squad"] = get_team_squad(id_away, api_key)
        else:
            result["home_squad"] = get_team_squad(id_home, api_key)
            result["away_squad"] = get_team_squad(id_away, api_key)
    except Exception as e:
        result["error"] = str(e)
        result.setdefault("sync_log", []).append(f"Error de connexió o API: {e}")

    return result
