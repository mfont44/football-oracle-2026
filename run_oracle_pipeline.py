# -*- coding: utf-8 -*-
"""
Pipeline mestre Oracle: executa tot el procés de dades i model amb un sol comandament.

Seqüència:
  1. integrate_espn_data.py        Micro-stats ESPN
  2. process_statsbomb.py          Event data / xG StatsBomb
  3. integrate_multi_league_2026.py  Dades històriques 5 grans lligues (LaLiga, Premier, Bundes, League1, SerieA) → multi_league_2026_summary.csv
  4. football_pro_model.py         Entrenament amb features Home/Away (usa el resum multi-lliga per ml_avg_corners, ml_avg_cards, etc.)
  5. prune_features.py             Poda de variables

En cas d'error, el procés s'atura i es mostra l'error.
Al final es pot llançar la interfície web (streamlit run app.py).
"""

from __future__ import annotations

import os
import subprocess
import sys

# Directori del projecte (on estan els scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("integrate_espn_data.py", "Micro-stats ESPN"),
    ("process_statsbomb.py", "Event data / xG StatsBomb"),
    ("integrate_multi_league_2026.py", "Dades històriques 5 grans lligues (LaLiga, Premier, Bundes, League1, SerieA)"),
    ("football_pro_model.py", "Entrenament model Home/Away"),
    ("prune_features.py", "Poda de variables"),
]


def run_step(step_num: int, script: str, description: str) -> bool:
    """Executa un script i retorna True si ha acabat amb èxit."""
    total = len(STEPS)
    print()
    print("=" * 60)
    print(f"  PAS {step_num} de {total}: {description}")
    print(f"  Comandament: python {script}")
    print("=" * 60)

    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.isfile(path):
        print(f"ERROR: No s'ha trobat el fitxer '{script}'.")
        return False

    try:
        result = subprocess.run(
            [sys.executable, path],
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print()
            print(f"ERROR: El script '{script}' ha acabat amb codi {result.returncode}.")
            return False
    except Exception as e:
        print()
        print(f"ERROR en executar '{script}': {e}")
        return False

    print()
    print(f"  PAS {step_num} de {total} completat.")
    return True


def main() -> None:
    print()
    print("  ORACLE PIPELINE — Procés complet de dades i model")
    print("  ==================================================")

    for i, (script, desc) in enumerate(STEPS, start=1):
        if not run_step(i, script, desc):
            print()
            print("El pipeline s'ha aturat. Corrigeix l'error i torna a executar.")
            sys.exit(1)

    print()
    print("=" * 60)
    print("  Tots els passos han acabat correctament.")
    print("=" * 60)

    # Pregunta per llançar la interfície web
    try:
        resp = input("\nVols llançar la interfície web ara? (y/n): ").strip().lower()
    except EOFError:
        resp = "n"

    if resp == "y" or resp == "s" or resp == "yes" or resp == "si":
        app_path = os.path.join(SCRIPT_DIR, "app.py")
        if not os.path.isfile(app_path):
            print(f"No s'ha trobat app.py a {SCRIPT_DIR}")
            sys.exit(1)
        print("\nIniciant Streamlit (app.py)...")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", "app.py"],
                cwd=SCRIPT_DIR,
            )
        except KeyboardInterrupt:
            print("\nStreamlit aturat.")
        except Exception as e:
            print(f"Error en llançar Streamlit: {e}")
            sys.exit(1)
    else:
        print("No s'ha llançat la interfície. Per fer-ho manualment: streamlit run app.py")


if __name__ == "__main__":
    main()
