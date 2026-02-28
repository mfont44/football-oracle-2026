# -*- coding: utf-8 -*-
"""
Dashboard premium betting: predicció Over 2.5 gols, gauge, H2H, heatmap Poisson, radar.
Tema fosc professional. Gauge 0–100% per P(Over 2.5), taula H2H últims 3 enfrontaments.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import football_pro_model as fpm
    try:
        import api_football as af
    except ImportError:
        af = None
    model_main = fpm.main
    predictor = fpm.predictor
    cercar_equip = fpm.cercar_equip
    llistat_equips = fpm.llistat_equips
    apply_live_adjustment = fpm.apply_live_adjustment
except ImportError as e:
    st.error(f"No s'ha pogut importar el model: {e}. Assegura't que football_pro_model.py és a la mateixa carpeta.")
    st.stop()

# ============== ESTÈTICA WEB (style.css + màxim contrast) ==============
# Integració: variables, Tailwind-like (bg-black, card, glow), overrides Streamlit
STYLE = """
<style>
  /* ---- Variables (estil style.css) ---- */
  :root {
    --bg-black: #000000;
    --bg-zinc-900: #18181b;
    --border-zinc-700: #3f3f46;
    --text-white: #FFFFFF;
    --text-black: #000000;
    --accent-green: #22c55e;
    --font-sans: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* ---- Base (fons negre, tipografia) ---- */
  [data-testid="stAppViewContainer"], .stApp, html, body { background: var(--bg-black) !important; }
  html, body, p, .stMarkdown {
    font-family: var(--font-sans) !important;
    color: var(--text-white);
  }
  h1, h2, h3, .stSubheader {
    font-family: var(--font-sans) !important;
    color: var(--text-white) !important;
  }

  /* ---- Màxim contrast: TOTS els botons = text negre, fons blanc ---- */
  .stButton > button,
  .stButton > button[kind="primary"],
  .stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 0.75rem !important;
    font-family: var(--font-sans) !important;
  }
  .stButton > button:hover,
  .stButton > button[kind="primary"]:hover,
  .stButton > button[kind="secondary"]:hover {
    background: #f4f4f5 !important;
    color: #000000 !important;
    box-shadow: 0 0 24px rgba(34, 197, 94, 0.3);
  }

  /* ---- Selectors: text blanc, fons negre, borde zinc-700 ---- */
  .stSelectbox label, .stCheckbox label, .stNumberInput label,
  label[data-testid="stWidgetLabel"] { color: #FFFFFF !important; font-family: var(--font-sans) !important; }
  div[data-testid="stSelectbox"] > div {
    background: #000000 !important;
    border: 2px solid var(--border-zinc-700) !important;
    border-radius: 0.5rem !important;
  }
  div[data-testid="stSelectbox"] > div:focus-within {
    border-color: #71717a !important;
    border-width: 2px !important;
  }
  div[data-testid="stSelectbox"] input,
  div[data-testid="stSelectbox"] span,
  div[data-testid="stSelectbox"] [role="combobox"] { color: #FFFFFF !important; }
  div[data-testid="stNumberInput"] input {
    background: #000000 !important;
    border: 2px solid var(--border-zinc-700) !important;
    color: #FFFFFF !important;
    border-radius: 0.5rem !important;
  }
  div[data-testid="stNumberInput"] input:focus {
    border-color: #71717a !important;
    box-shadow: none !important;
  }

  /* ---- Cards (Tailwind-like) ---- */
  .card {
    background: var(--bg-zinc-900);
    border: 1px solid var(--border-zinc-700);
    border-radius: 0.75rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
  }
  .stMetric, [data-testid="stVerticalBlock"] > div {
    background: var(--bg-zinc-900) !important;
    border: 1px solid var(--border-zinc-700) !important;
    border-radius: 0.75rem;
    padding: 1rem;
  }
  .stMetric label, .stMetric [data-testid="stMetricValue"] { color: #FFFFFF !important; }

  /* ---- Header (estil web) ---- */
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border-zinc-700);
    margin-bottom: 1.5rem;
    font-family: var(--font-sans);
  }
  .app-header .logo { font-size: 1.25rem; font-weight: 700; color: var(--text-white); }
  .app-header .tagline { font-size: 0.875rem; color: #a1a1aa; }

  /* ---- Hero: gradient gris → blanc ---- */
  .hero {
    text-align: center;
    margin-bottom: 2rem;
    font-family: var(--font-sans);
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
    background: linear-gradient(180deg, #ffffff 0%, #a1a1aa 50%, #71717a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero-subtitle {
    font-size: 1rem;
    color: #E5E5E5;
  }

  /* ---- Glow (style.css) ---- */
  .glow {
    box-shadow: 0 0 24px rgba(34, 197, 94, 0.25);
    border: 1px solid var(--border-zinc-700);
    border-radius: 0.75rem;
    background: var(--bg-zinc-900);
    padding: 1.25rem;
  }
  .verdict-box {
    background: var(--bg-zinc-900);
    padding: 1.25rem;
    border-radius: 0.75rem;
    border: 1px solid var(--border-zinc-700);
    color: #FFFFFF !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8);
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
  }
  .h2h-card {
    background: var(--bg-zinc-900);
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    border: 1px solid var(--border-zinc-700);
    margin: 0.25rem 0;
    color: #FFFFFF;
  }
  .team-header { font-size: 1.5rem; font-weight: 600; color: #FFFFFF; text-align: center; margin: 0.5rem 0; font-family: var(--font-sans); }
  .crest { font-size: 2.5rem; text-align: center; margin-bottom: 0.25rem; }
  .alerta-sorpresa { background: var(--bg-zinc-900); padding: 1rem; border-radius: 0.75rem; border: 1px solid var(--border-zinc-700); margin: 1rem 0; color: #E5E5E5; }
  .badge-generic { display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.75rem; margin-left:0.5rem; color: #E5E5E5; border:1px solid var(--border-zinc-700); }

  /* ---- Animació fade-in-up (style.css) ---- */
  .fade-in-up {
    animation: fadeInUp 0.5s ease-out forwards;
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .result-block { animation: fadeInUp 0.5s ease-out forwards; }

  [data-testid="stSidebar"] { background: var(--bg-black) !important; border-right: 1px solid var(--border-zinc-700); }
  .stExpander { border: 2px solid var(--border-zinc-700); border-radius: 0.75rem; background: var(--bg-zinc-900); }
  .stExpander label { color: #FFFFFF !important; font-family: var(--font-sans) !important; }
</style>
"""

# Capa HTML: Header + Hero (estil web / Tailwind-like)
HEADER_HTML = """
<header class="app-header">
  <div class="logo">⚽ Football Oracle</div>
  <span class="tagline">Over 2.5 · Poisson · XGBoost</span>
</header>
"""
HERO_HTML = """
<div class="hero">
  <div class="hero-title">Football Oracle 2026. Predict the unpredictable.</div>
  <div class="hero-subtitle">Advanced AI-driven insights for Over 2.5 goals and exact scores.</div>
</div>
"""
@st.cache_resource
def get_model_and_teams():
    """Model, llistat d'equips i mètriques (cachejat). Prioritza .pkl pre-entrenats per evitar timeout 503."""
    import os
    import joblib

    pkl_model = "football_model.pkl"
    pkl_cols = "feature_cols.pkl"
    pkl_games = "games_full.pkl"

    if os.path.isfile(pkl_model) and os.path.isfile(pkl_cols) and os.path.isfile(pkl_games):
        with st.spinner("Carregant model pre-entrenat..."):
            fpm.model = joblib.load(pkl_model)
            fpm.feature_cols = joblib.load(pkl_cols)
            fpm.games_full = joblib.load(pkl_games)
            fpm.clubs_df = fpm.games_full.get("clubs")
            noms = fpm.llistat_equips(fpm.clubs_df) if fpm.clubs_df is not None else []
            return fpm.model, noms, None, None, None, None

    with st.spinner("Carregant dades i entrenant el model (només la primera vegada)..."):
        model, _df, _, acc, conf_mat, imp_df, base_acc = model_main()
        noms = fpm.llistat_equips(fpm.clubs_df) if fpm.clubs_df is not None else []
        return model, noms, acc, conf_mat, imp_df, base_acc


def format_millions(value: float) -> str:
    if value >= 1e9:
        return f"€{value/1e9:.2f}B"
    if value >= 1e6:
        return f"€{value/1e6:.1f}M"
    if value >= 1e3:
        return f"€{value/1e3:.1f}K"
    return f"€{value:.0f}"


def plot_gauge_over25(over_25_prob: float, height: int = 200) -> go.Figure:
    """Gauge Dark/Apple: blanc–gris; accent verd quan probabilitat alta (Over 2.5)."""
    bar_color = "#22c55e" if over_25_prob >= 55 else "#e4e4e7"
    threshold_color = "#22c55e" if over_25_prob >= 55 else "#ffffff"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=over_25_prob,
        number=dict(suffix="%", font=dict(size=22, color="#ffffff")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickfont=dict(color="#FFFFFF", size=10)),
            bar=dict(color=bar_color),
            bgcolor="rgba(24, 24, 27, 0.8)",
            borderwidth=1,
            bordercolor="#3f3f46",
            steps=[],
            threshold=dict(line=dict(color=threshold_color, width=2), thickness=0.75, value=over_25_prob),
        ),
        title=dict(text="P(Over 2.5 gols)", font=dict(size=14, color="#FFFFFF")),
    ))
    fig.update_layout(
        margin=dict(t=35, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff", family="Inter, Segoe UI, sans-serif"),
        height=height,
    )
    return fig


def plot_heatmap_scores(score_matrix: np.ndarray) -> go.Figure:
    """Heatmap Poisson: escala zinc → blanc (Dark/Apple)."""
    mat = np.asarray(score_matrix)
    n = mat.shape[0]
    z_plot = mat.T * 100
    fig = go.Figure(data=go.Heatmap(
        z=z_plot,
        x=[str(i) for i in range(n)],
        y=[str(j) for j in range(n)],
        colorscale=[[0, "#18181b"], [0.5, "#3f3f46"], [1, "#fafafa"]],
        text=np.round(z_plot, 1),
        texttemplate="%{text}%",
        textfont=dict(size=10, color="#fafafa"),
        hovertemplate="Local %{x} - Visitant %{y}: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Gols local",
        yaxis_title="Gols visitant",
        xaxis=dict(side="bottom", tickfont=dict(color="#FFFFFF"), gridcolor="#3f3f46", showgrid=True),
        yaxis=dict(tickfont=dict(color="#FFFFFF"), autorange="reversed", gridcolor="#3f3f46", showgrid=True),
        margin=dict(t=30, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", family="Inter, Segoe UI, sans-serif"),
        height=380,
    )
    return fig


def plot_radar(
    home_gf: float, home_ga: float, home_mv: float, home_corners: float, home_cards: float,
    away_gf: float, away_ga: float, away_mv: float, away_corners: float, away_cards: float,
    nom_local: str, nom_visitant: str,
) -> go.Figure:
    """Radar Dark/Apple: blanc i zinc només."""
    max_gf = max(home_gf, away_gf, 0.01)
    max_ga = max(home_ga, away_ga, 0.01)
    max_mv = max(home_mv, away_mv, 1.0)
    max_corners = max(home_corners, away_corners, 0.01)
    max_cards = max(home_cards, away_cards, 0.01)
    r_home = [
        home_gf / max_gf,
        1 - (home_ga / max_ga),
        home_mv / max_mv,
        home_corners / max_corners,
        home_cards / max_cards,
    ]
    r_away = [
        away_gf / max_gf,
        1 - (away_ga / max_ga),
        away_mv / max_mv,
        away_corners / max_corners,
        away_cards / max_cards,
    ]
    categories = ["Atac", "Defensa", "Valor de mercat", "Córners", "Targetes"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_home + [r_home[0]], theta=categories + [categories[0]],
        fill="toself", name=nom_local[:20],
        line=dict(color="#fafafa", width=2),
        fillcolor="rgba(250, 250, 250, 0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_away + [r_away[0]], theta=categories + [categories[0]],
        fill="toself", name=nom_visitant[:20],
        line=dict(color="#71717a", width=2),
        fillcolor="rgba(113, 113, 122, 0.15)",
    ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(24, 24, 27, 0.9)", radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#FFFFFF"), gridcolor="#3f3f46")),
        showlegend=True,
        legend=dict(font=dict(color="#FFFFFF", family="Inter, Segoe UI, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", family="Inter, Segoe UI, sans-serif"),
        height=380,
    )
    return fig


def plot_real_vs_potential(
    home_poss: float,
    home_shots_on: float,
    home_mv: float,
    away_poss: float,
    away_shots_on: float,
    away_mv: float,
    nom_local: str,
    nom_visitant: str,
) -> go.Figure:
    """
    Comparativa barres:
    - Rendiment Real 2025: possessió mitjana ESPN + tirs a porteria.
    - Potencial Històric: valor de mercat (Transfermarkt).
    """
    # Escalat 0-1 per fer comparables les magnituds
    max_poss = max(home_poss, away_poss, 1.0)
    max_shots = max(home_shots_on, away_shots_on, 1.0)
    max_mv = max(home_mv, away_mv, 1.0)

    home_real = (home_poss / max_poss + home_shots_on / max_shots) / 2.0
    away_real = (away_poss / max_poss + away_shots_on / max_shots) / 2.0
    home_potential = home_mv / max_mv
    away_potential = away_mv / max_mv

    categories = ["Rendiment Real 2025", "Potencial Històric"]
    x = [f"{nom_local[:18]}", f"{nom_local[:18]}", f"{nom_visitant[:18]}", f"{nom_visitant[:18]}"]
    y = [home_real, home_potential, away_real, away_potential]
    color = ["#fafafa", "#71717a", "#fafafa", "#71717a"]
    text = [
        "Real",
        "Potencial",
        "Real",
        "Potencial",
    ]

    fig = go.Figure(
        data=go.Bar(
            x=x,
            y=y,
            marker_color=color,
            text=text,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Rendiment Real 2025 vs Potencial Històric",
        yaxis_title="Índex normalitzat (0-1)",
        xaxis_title="Equip",
        yaxis=dict(range=[0, 1.1], tickfont=dict(color="#FFFFFF"), gridcolor="#3f3f46", showgrid=True),
        xaxis=dict(tickfont=dict(color="#FFFFFF"), gridcolor="#3f3f46", showgrid=True),
        margin=dict(t=60, b=40, l=60, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", family="Inter, Segoe UI, sans-serif"),
        height=360,
    )
    return fig


def run_app() -> None:
    import os
    st.set_page_config(page_title="Predicció Over 2.5", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)

    try:
        _model, noms_equips, acc, conf, imp_df, baseline_acc = get_model_and_teams()
    except FileNotFoundError as e:
        st.error(f"Dades no trobades: {e}")
        return
    except Exception as e:
        st.error(f"Error en carregar el model: {e}")
        return

    if _model is None or (hasattr(fpm, "model") and fpm.model is None):
        st.error("**Error: Model no disponible. Revisa els fitxers .pkl.**")
        return

    st.markdown(HEADER_HTML, unsafe_allow_html=True)
    st.markdown(HERO_HTML, unsafe_allow_html=True)

    with st.expander("Mètriques de rendiment del model"):
        if acc is not None:
            if baseline_acc is not None:
                delta = (acc - baseline_acc) * 100
                st.metric("Accuracy", f"{acc * 100:.1f}%", f"{delta:+.2f} pts vs. baseline")
            else:
                st.metric("Accuracy", f"{acc * 100:.1f}%")
        if conf is not None and conf.size > 0:
            labels = ["Under 2.5", "Over 2.5"] if conf.shape[0] == 2 else ["X (empat)", "1 (local)", "2 (visitant)"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=conf,
                x=labels,
                y=labels,
                colorscale=[[0, "#18181b"], [0.5, "#3f3f46"], [1, "#fafafa"]],
                text=conf.astype(int),
                texttemplate="%{text}",
                textfont=dict(size=14, color="#fafafa"),
                hovertemplate="Real: %{y} | Predit: %{x} = %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                title="Matriu de confusió",
                xaxis_title="Predit",
                yaxis_title="Real",
                xaxis=dict(tickfont=dict(color="#FFFFFF"), gridcolor="#3f3f46"),
                yaxis=dict(tickfont=dict(color="#FFFFFF"), autorange="reversed", gridcolor="#3f3f46"),
                margin=dict(t=40, b=40, l=80, r=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFFFFF", family="Inter, Segoe UI, sans-serif"),
                height=360,
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        if imp_df is not None and not imp_df.empty:
            top10 = imp_df.head(10)
            fig_imp = go.Figure(data=go.Bar(
                x=top10["importancia"],
                y=top10["variable"],
                orientation="h",
                marker_color="#71717a",
                text=[f"{v:.3f}" for v in top10["importancia"]],
                textposition="outside",
                textfont=dict(color="#fafafa"),
            ))
            fig_imp.update_layout(
                title="Top 10 Feature Importance",
                xaxis_title="Importància",
                yaxis_title="Variable",
                xaxis=dict(tickfont=dict(color="#FFFFFF"), gridcolor="#3f3f46", showgrid=True),
                yaxis=dict(tickfont=dict(color="#FFFFFF"), autorange="reversed", gridcolor="#3f3f46", showgrid=True),
                margin=dict(t=40, b=40, l=120, r=80),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFFFFF", family="Inter, Segoe UI, sans-serif"),
                height=400,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        if acc is None and conf is None and (imp_df is None or imp_df.empty):
            pass

    def _default_index(prefix: str) -> int:
        for i, nom in enumerate(noms_equips):
            if prefix.lower() in nom.lower():
                return i
        return 0

    # Card form (estil web / Tailwind-like)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Selectors al cos principal: dues columnes amples
    col_local, col_away = st.columns(2)
    with col_local:
        nom_local = st.selectbox("Equip local", options=noms_equips, index=_default_index("Real Madrid"), key="local")
    with col_away:
        nom_visitant = st.selectbox("Equip visitant", options=noms_equips, index=_default_index("Barcelona"), key="away")

    with st.expander("Eliminatòria / Anada", expanded=False):
        is_knockout = st.checkbox("És eliminatòria", value=False, key="is_knockout")
        is_return_leg = st.checkbox("És partit de tornada", value=False, key="is_return_leg")
        first_leg_diff = st.number_input("Diferència anada (local − visitant)", min_value=-10, max_value=10, value=0, step=1, key="first_leg_diff")

    run_clicked = st.button("Predir", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    api_key = os.environ.get("FOOTBALL_DATA_ORG_KEY", os.environ.get("API_FOOTBALL_KEY", "LA_TEVA_KEY_AQUÍ"))
    try:
        if hasattr(st, "secrets") and st.secrets.get("api_football_key"):
            api_key = st.secrets["api_football_key"]
    except Exception:
        pass  # sense secrets.toml es fa servir la variable d'entorn o el valor per defecte


    if run_clicked:
        st.session_state.pop("live_adjusted_result", None)
        st.session_state.pop("baixes_detectades", None)
        result: Optional[dict[str, Any]] = None
        if fpm.model is None:
            st.error("**Error: Model no disponible. Revisa els fitxers .pkl.**")
        else:
            try:
                result = predictor(nom_local, nom_visitant, verbose=False, is_knockout=is_knockout, is_return_leg=is_return_leg, first_leg_diff=float(first_leg_diff))
            except Exception as e:
                st.error(f"Error en la predicció: {e}")
                import traceback
                with st.expander("Detall de l'error (debug)"):
                    st.code(traceback.format_exc())
            else:
                if result is None:
                    id_l, sugg_l = cercar_equip(nom_local, fpm.clubs_df) if fpm.clubs_df is not None else (None, [])
                    id_a, sugg_a = cercar_equip(nom_visitant, fpm.clubs_df) if fpm.clubs_df is not None else (None, [])
                    if id_l is None and sugg_l:
                        st.warning(f"Equip no trobat: **{nom_local}**. Suggeriments: {', '.join(sugg_l[:3])}")
                    if id_a is None and sugg_a:
                        st.warning(f"Equip no trobat: **{nom_visitant}**. Suggeriments: {', '.join(sugg_a[:3])}")
                else:
                    st.session_state["last_result"] = result
                    st.session_state["last_nom_local"] = nom_local
                    st.session_state["last_nom_visitant"] = nom_visitant
                    st.toast("Predicció calculada.", icon="✅")

    if not st.session_state.get("last_result"):
        st.info("Tria equip local i visitant i prem **Predir**.")
        return

    # Resultats premium: dins div amb fade-in-up + glow (style.css)
    if st.session_state.get("last_result"):
        st.markdown('<div class="fade-in-up glow result-block">', unsafe_allow_html=True)
        display_result = st.session_state.get("live_adjusted_result") or st.session_state["last_result"]
        nom_local_c = display_result.get("nom_local", st.session_state.get("last_nom_local", ""))
        nom_visitant_c = display_result.get("nom_visitant", st.session_state.get("last_nom_visitant", ""))

        # Capçalera: escuts (emoji) + noms
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown('<div class="crest">🏠</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-header">{nom_local_c}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="crest">⚽</div>', unsafe_allow_html=True)
            st.markdown('<div class="team-header">vs</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="crest">✈️</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-header">{nom_visitant_c}</div>', unsafe_allow_html=True)

        # Advertència si un equip no es troba al CSV (dades genèriques)
        if display_result.get("generic_team_local"):
            st.warning(f"**{nom_local_c}** no es troba al dataset Transfermarkt. S’estan usant dades genèriques (mitjana global) per a Rolling Averages i Valor de mercat.")
        if display_result.get("generic_team_away"):
            st.warning(f"**{nom_visitant_c}** no es troba al dataset Transfermarkt. S’estan usant dades genèriques (mitjana global) per a Rolling Averages i Valor de mercat.")

        # Mètriques: valor de mercat
        home_mv = display_result.get("home_mv", 0)
        away_mv = display_result.get("away_mv", 0)
        home_espn_has = bool(display_result.get("home_espn_has_data", False))
        away_espn_has = bool(display_result.get("away_espn_has_data", False))
        m1, m2 = st.columns(2)
        with m1:
            label_local = "Valor de mercat (local)"
            if not home_espn_has:
                label_local += "  "
            st.metric(label_local, format_millions(home_mv))
            if not home_espn_has:
                st.markdown(
                    '<span class="badge-generic">ESPN genèric · forma estimada</span>',
                    unsafe_allow_html=True,
                )
        with m2:
            label_away = "Valor de mercat (visitant)"
            if not away_espn_has:
                label_away += "  "
            st.metric(label_away, format_millions(away_mv))
            if not away_espn_has:
                st.markdown(
                    '<span class="badge-generic">ESPN genèric · forma estimada</span>',
                    unsafe_allow_html=True,
                )

        st.subheader("Resultat 1-X-2 (Poisson)")
        p_1 = display_result.get("p_1", 0)
        p_x = display_result.get("p_x", 0)
        p_2 = display_result.get("p_2", 0)
        col_1, col_x, col_2 = st.columns(3)
        with col_1:
            st.metric("1 · Victòria local", f"{float(p_1):.1f}%", help="Probabilitat que guanyi l'equip local")
        with col_x:
            st.metric("X · Empat", f"{float(p_x):.1f}%", help="Probabilitat d'empat")
        with col_2:
            st.metric("2 · Victòria visitant", f"{float(p_2):.1f}%", help="Probabilitat que guanyi l'equip visitant")

        st.subheader("Probabilitat de gols")
        over_05 = display_result.get("over_05_prob", 0)
        over_15 = display_result.get("over_15_prob", 0)
        over_25_prob = display_result.get("over_25_prob", 0)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.metric("+0.5 Gols", f"{float(over_05):.1f}%", "Almenys 1 gol", help="Derivat de Poisson")
        with g2:
            st.metric("+1.5 Gols", f"{float(over_15):.1f}%", "Almenys 2 gols", help="Derivat de Poisson")
        with g3:
            st.metric("+2.5 Gols", f"{float(over_25_prob):.1f}%", "Almenys 3 gols", help="Model XGBoost (57.2% accuracy)")

        # Fila 1: Gauge (petit) + Verdict
        st.subheader("Probabilitat Over 2.5 gols")
        row1_col1, row1_col2 = st.columns([1, 1])
        with row1_col1:
            st.plotly_chart(plot_gauge_over25(float(over_25_prob), height=200), use_container_width=True)
        with row1_col2:
            st.markdown(
                f'<div class="verdict-box">'
                f'<strong>🏆 Verdict</strong><br><br>'
                f'El model assigna un <strong>{float(over_25_prob):.1f}%</strong> de probabilitat a més de 2.5 gols.'
                f'</div>',
                unsafe_allow_html=True,
            )
        h2h_matches = display_result.get("h2h_matches") or []
        st.subheader("Històric directe (H2H)")
        if h2h_matches:
            for m in h2h_matches:
                st.markdown(
                    f'<div class="h2h-card"><strong>{m.get("season", "?")}</strong>: {m.get("score", "?")} · '
                    f'{"Over 2.5" if m.get("over_25") else "Under 2.5"}</div>',
                    unsafe_allow_html=True,
                )

        # Alerta de Sorpresa: favorit per valor però rival amb millor ratxa (EWM)
        home_gf = display_result.get("home_roll_gf", 0)
        away_gf = display_result.get("away_roll_gf", 0)
        home_ga = display_result.get("home_roll_ga", 0)
        away_ga = display_result.get("away_roll_ga", 0)
        surprise = False
        if home_mv > away_mv * 1.3 and away_gf > home_gf * 1.25:
            surprise = True
            msg = f"⚠️ **Alerta de Sorpresa:** {nom_visitant_c} té menys valor de mercat però una ratxa (gols a favor) molt superior a la del favorit. El resultat pot ser més obert del que indica el valor."
        elif away_mv > home_mv * 1.3 and home_gf > away_gf * 1.25:
            surprise = True
            msg = f"⚠️ **Alerta de Sorpresa:** {nom_local_c} té menys valor de mercat però una ratxa (gols a favor) molt superior a la del favorit. El resultat pot ser més obert del que indica el valor."
        if surprise:
            st.markdown(f'<div class="alerta-sorpresa">{msg}</div>', unsafe_allow_html=True)

        st.subheader("Marcadors exactes (Poisson)")
        mat = display_result.get("score_matrix")
        top3 = display_result.get("top3_exact", [])
        if top3:
            ex1, ex2, ex3 = st.columns(3)
            for col, (score, prob) in zip([ex1, ex2, ex3], top3):
                with col:
                    st.markdown(
                        f'<div class="h2h-card" style="text-align:center; padding:1.2rem;">'
                        f'<div style="font-size:2rem; font-weight:700; color:#ffffff;">{score}</div>'
                        f'<div style="color:#E5E5E5; margin-top:0.3rem;">{float(prob):.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        # Fila 2: Radar + Heatmap Poisson (un al costat de l'altre)
        home_gf_short = float(display_result.get("home_roll_gf_short", 0) or display_result.get("home_roll_gf", 0))
        home_gf_long = float(display_result.get("home_roll_gf_long", 0) or home_gf_short)
        away_gf_short = float(display_result.get("away_roll_gf_short", 0) or display_result.get("away_roll_gf", 0))
        away_gf_long = float(display_result.get("away_roll_gf_long", 0) or away_gf_short)
        home_attack = 0.7 * home_gf_short + 0.3 * home_gf_long
        away_attack = 0.7 * away_gf_short + 0.3 * away_gf_long
        fig_r = plot_radar(
            home_attack, display_result.get("home_roll_ga", 0), display_result.get("home_mv", 0),
            float(display_result.get("home_corners_metric", 0) or 0), float(display_result.get("home_cards_metric", 0) or 0),
            away_attack, display_result.get("away_roll_ga", 0), display_result.get("away_mv", 0),
            float(display_result.get("away_corners_metric", 0) or 0), float(display_result.get("away_cards_metric", 0) or 0),
            nom_local_c, nom_visitant_c,
        )
        fig_h = plot_heatmap_scores(np.asarray(mat)) if mat is not None else None
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.subheader("Comparativa (Radar)")
            st.plotly_chart(fig_r, use_container_width=True)
        with row2_col2:
            st.subheader("Heatmap Poisson")
            if fig_h is not None:
                st.plotly_chart(fig_h, use_container_width=True)
        home_acc = float(display_result.get("home_form_acceleration", 0) or 0)
        away_acc = float(display_result.get("away_form_acceleration", 0) or 0)
        acc_indicators = []
        if home_acc > 0:
            acc_indicators.append(f'<span style="color:#E5E5E5;">↑ {nom_local_c}: ratxa en alça</span>')
        if away_acc > 0:
            acc_indicators.append(f'<span style="color:#E5E5E5;">↑ {nom_visitant_c}: ratxa en alça</span>')
        if acc_indicators:
            st.markdown(" ".join(acc_indicators))

        # Nou gràfic: Rendiment Real 2025 (ESPN) vs Potencial Històric (Valor de Mercat)
        home_poss_espn = float(display_result.get("home_espn_avg_possession", 0) or 0)
        away_poss_espn = float(display_result.get("away_espn_avg_possession", 0) or 0)
        home_shots_espn = float(display_result.get("home_espn_avg_shots_on_target", 0) or 0)
        away_shots_espn = float(display_result.get("away_espn_avg_shots_on_target", 0) or 0)

        st.subheader("Rendiment Real 2025 vs Potencial Històric")
        if (home_poss_espn > 0 or away_poss_espn > 0) and (home_shots_espn > 0 or away_shots_espn > 0):
            fig_real_pot = plot_real_vs_potential(
                home_poss_espn,
                home_shots_espn,
                home_mv,
                away_poss_espn,
                away_shots_espn,
                away_mv,
                nom_local_c,
                nom_visitant_c,
            )
            st.plotly_chart(fig_real_pot, use_container_width=True)
        else:
            pass

        # Botó Comprovar baixes (API) + secció Baixes detectades
        @st.cache_data(ttl=3600)
        def _fetch_live_data(home_name: str, away_name: str, key: str):
            try:
                return af.get_live_data_for_match(home_name, away_name, key)
            except Exception:
                return {"home_injuries": [], "away_injuries": [], "error": "Error de connexió"}

        if st.button("🔄 Sincronitzar amb Football-Data.org i Ajustar", use_container_width=True):
            if af is None:
                st.warning("Mòdul **api_football** no trobat. Afegeix api_football.py a la carpeta del projecte.")
            else:
                home_name = (st.session_state.get("last_nom_local") or "").strip()
                away_name = (st.session_state.get("last_nom_visitant") or "").strip()
                if not home_name or not away_name:
                    st.warning("Falten noms d’equips per sincronitzar. Feu primer una predicció amb equips vàlids.")
                else:
                    with st.status("Sincronitzant amb Football-Data.org...", expanded=True) as status:
                        st.write("Consultant partits i alineacions...")
                        live_data = _fetch_live_data(home_name, away_name, api_key)
                        if live_data.get("error"):
                            status.update(label="Error de connexió", state="error")
                            st.error(
                                f"**Error:** {live_data.get('error')}. "
                                "Comprova la clau X-Auth-Token. Es manté la predicció inicial."
                            )
                        else:
                            status.update(label="Sincronització completada", state="complete")
                            sync_log = live_data.get("sync_log") or []
                            if sync_log:
                                with st.expander("Log de matching", expanded=False):
                                    for line in sync_log:
                                        st.text(line)
                        adjusted = apply_live_adjustment(st.session_state["last_result"], live_data)
                        st.session_state["live_adjusted_result"] = adjusted
                    st.toast("Predicció ajustada amb dades en viu.", icon="✅")
                    st.rerun()

        baixes = display_result.get("baixes_detectades", [])
        probs_init = display_result.get("probs_initial")
        probs_adj = display_result.get("probs_adjusted")
        if baixes and probs_init and probs_adj:
            st.subheader("📋 Baixes detectades i ajust Over 2.5")
            for b in baixes:
                equip = "Local" if b.get("equip") == "local" else "Visitant"
                st.markdown(f"- **{equip}:** {b.get('nom', '?')} ({b.get('reason', '')})")
            p_i = probs_init.get("over_25_prob", 0)
            p_a = probs_adj.get("over_25_prob", 0)
            st.markdown(f"**P(Over 2.5):** {p_i:.1f}% → **{p_a:.1f}%** (penalitzat per baixes)")
        elif display_result.get("probs_initial") and not baixes and st.session_state.get("live_adjusted_result"):
            st.info("S’han consultat les dades en viu; no s’han detectat jugadors clau absents.")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
