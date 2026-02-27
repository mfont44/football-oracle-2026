# ⚽ Football Oracle Pro

**Sistema de predicció de futbol amb model híbrid (CSV + API) i interfície premium per a mercats Over/Under.**

---

## 📋 Descripció

**Football Oracle Pro** és una aplicació de predicció de resultats que combina:

- **Dades històriques** (Transfermarkt: partits, plantilles, valor de mercat, jugadors).
- **Micro-stats i xG** (ESPN: possessió, tirs; StatsBomb: expected goals, key passes → tactical_danger_index).
- **Dades multi-lliga** (córners, targetes, expectativa de mercat per a lligues actuals).
- **Dades en temps quasi real** (Football-Data.org: alineacions, baixes, partits).
- **Model XGBoost** entrenat per predir **Over 2.5 gols** amb **ratxes per camp (Home/Away)**, advantage casa–fora i **ajust en viu** quan l’API detecta baixes de jugadors clau.

L’app ofereix un **dashboard tipus betting**: velocímetre de probabilitat Over 2.5, històric directe (H2H), heatmap de marcadors (Poisson) i comparativa d’equips (radar), amb opció de sincronitzar amb Football-Data.org per penalitzar la probabilitat si falten davanters titulars.

---

## 🚀 Pipeline mestre (un sol comandament)

Es pot executar tot el flux de dades i entrenament amb:

```bash
python run_oracle_pipeline.py
```

**Seqüència d’execució:**

| Pas | Script | Descripció |
|-----|--------|------------|
| 1 | `integrate_espn_data.py` | Micro-stats ESPN (possessió, tirs, índex ofensiu/defensiu). |
| 2 | `process_statsbomb.py` | Event data / xG StatsBomb (tactical_danger_index). |
| 3 | `integrate_multi_league_2026.py` | Dades lligues actuals (córners, targetes, market_expectation). |
| 4 | `football_pro_model.py` | Entrenament del model amb features Home/Away. |
| 5 | `prune_features.py` | Poda automàtica de variables (importància &lt; umbral). |

- Si un script falla, el procés s’atura i es mostra l’error.
- Al final, si la poda ha acabat bé, l’script pregunta: **«Vols llançar la interfície web ara? (y/n)»**; amb `y` s’executa `streamlit run app.py`.

---

## 🎯 Estat actual del projecte

| Àmbit | Detall |
|--------|--------|
| **Model** | XGBoost binari (`binary:logistic`) per **Over 2.5 gols**. Entrenament amb split cronològic (temporada test 2024 o 20% més recent). |
| **Accuracy** | Mètrica principal: accuracy en conjunt de test. Poda de features per eliminar soroll (umbral d’importància). |
| **Dades històriques** | CSVs Transfermarkt: `clubs`, `club_games`, `games`, `players` (Squad Value per `current_club_id`). |
| **Dades en viu** | API Football-Data.org (X-Auth-Token): partits, alineacions/squad per detectar jugadors clau absents. |
| **Features del model** | Ratxes **per camp (Home/Away)**, Home/Away Advantage, EWM short/long, Market Value Ratio, nou entrenador, dies de descans, H2H, ESPN, StatsBomb, multi-league. |

### Features implementades

| Feature | Descripció |
|--------|------------|
| **Ratxes per camp (doble EWM)** | Forma segons on juga l’equip: `roll_gf_home` / `roll_gf_away` (gols a favor a casa / fora), mateix per gols en contra i punts. Finestres **short (4)** i **long (10)**. |
| **Home/Away Advantage** | Diferència rendiment a casa vs a fora (punts, gols a favor, gols en contra). Permet detectar equips que depenen molt del factor camp. |
| **Market Value Ratio** | Proporció valor de mercat local / visitant (Squad Value des de `players.csv`). |
| **Efecte nou entrenador** | Indicador binari si l’entrenador ha canviat respecte al partit anterior (`own_manager_name` a `club_games`). |
| **Dies de descans** | Dies naturals des de l’últim partit de cada equip (`home_days_rest`, `away_days_rest`). |
| **H2H històric** | Últims enfrontaments directes: `h2h_avg_goals` i `h2h_over25_rate`. Fallback: mitjana de la lliga. |
| **ESPN (micro-stats)** | Possessió, tirs a porteria, faltes, córners; índex ofensiu i defensiu per equip (resum a `csvfiles/espn_2025_summary.csv`). |
| **StatsBomb** | xG mitjà i key passes → **tactical_danger_index** per equip/temporada (si existeix `csvfiles/statsbomb_summary.csv`). |
| **Multi-league 2026** | Córners mitjans, targetes, market_expectation per a lligues actuals. |
| **Poda de variables** | Després de l’entrenament, `prune_features.py` elimina variables amb importància &lt; 0.005; la llista podada es guarda a `csvfiles/features_pruned.txt` i el model la utilitza en la propera execució. |

---

## 🖥️ Funcionalitats de la UI

- **Velocímetre de probabilitat (Gauge)**  
  Indicador 0–100% per **P(Over 2.5 gols)** amb escala tipus velocímetre (Plotly).

- **Verdict i nota**  
  Resum de la probabilitat Over 2.5 i nota amb dies de descans de cada equip.

- **Històric directe (H2H)**  
  Targetes amb els últims 3 enfrontaments (temporada, marcador, Over/Under 2.5). Si no n’hi ha, es mostra el missatge de fallback (mitjana lliga).

- **Marcador de Poisson**  
  Heatmap de probabilitats de marcador exacte (model Poisson bivariat amb lambdes derivats de l’EWM i el valor relatiu). Top 3 marcadors més probables.

- **Heatmap de resultats**  
  Matriu de probabilitat Local × Visitant (gols) per visualitzar on es concentren les prediccions.

- **Comparativa de força (radar)**  
  Comparativa atac (gols a favor), defensa (menys gols en contra) i valor de mercat entre local i visitant.

- **Mètriques del model**  
  Expander amb **Accuracy actual** (darrer test), **matriu de confusió** (Under 2.5 / Over 2.5) i **Top 10 Feature Importance**.

- **Sincronització Football-Data.org**  
  Botó per actualitzar alineacions/baixes; si es detecten jugadors clau absents, es penalitza **P(Over 2.5)** i es mostra el canvi (probs inicial → ajustada).

---

## 📦 Instal·lació

### 1. Clonar o descomprimir el projecte

```bash
cd ezmoney
```

### 2. Entorn virtual (recomanat)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Instal·lar dependències

```bash
pip install -r requirements.txt
```

Dependències principals:

| Paquet | Versió mínima | Ús |
|--------|----------------|-----|
| pandas | 1.5.0 | Dades i preprocessat |
| numpy | 1.21.0 | Càlculs numèrics |
| scikit-learn | 1.0.0 | Mètriques, train/test |
| xgboost | 1.6.0 | Model binari Over 2.5 |
| scipy | 1.9.0 | Poisson |
| streamlit | 1.28.0 | App web |
| plotly | 5.14.0 | Gauge, heatmaps, radar |

### 4. Configurar la clau de l’API (Football-Data.org)

L’app accepta la clau de dues maneres:

**Opció A – Secrets de Streamlit (recomanat per desplegament)**

Crea la carpeta `.streamlit` al directori del projecte i dins un fitxer `secrets.toml`:

```toml
api_football_key = "LA_TEVA_X_AUTH_TOKEN"
```

**Opció B – Variable d’entorn**

```bash
# Windows (PowerShell)
$env:FOOTBALL_DATA_ORG_KEY = "LA_TEVA_X_AUTH_TOKEN"

# Linux/macOS
export FOOTBALL_DATA_ORG_KEY="LA_TEVA_X_AUTH_TOKEN"
```

Si no hi ha clau configurada, l’app funciona igual amb les dades històriques; la sincronització amb Football-Data.org i l’ajust per baixes quedarà desactivat o amb missatge d’avís.

### 5. Executar tot el pipeline (recomanat la primera vegada)

Per generar micro-stats ESPN, resum StatsBomb, dades multi-lliga, entrenar el model i podar variables:

```bash
python run_oracle_pipeline.py
```

Si tot acaba bé, l’script et preguntarà si vols llançar la interfície web (`y`/`n`).

### 6. Executar només l’app

```bash
streamlit run app.py
```

Obrir el navegador a la URL que indiqui Streamlit (normalment `http://localhost:8501`).

### 7. Dades CSV i fitxers de suport

**Obligatoris** (a `csvfiles/` o `csv_files/`):

- `clubs.csv`
- `club_games.csv`
- `games.csv`
- `players.csv`

**Opcionals** (milloren el model si existeixen):

- **ESPN:** `csvfiles/base_data/` amb `teamStats.csv`, `fixtures.csv`, etc. → genera `espn_2025_summary.csv`.
- **StatsBomb:** fitxers/CSV d’event data → `process_statsbomb.py` genera `csvfiles/statsbomb_summary.csv`.
- **Multi-league:** fonts configurades a `integrate_multi_league_2026.py` → dades de córners, targetes, expectativa.
- **Poda:** després d’entrenar, `csvfiles/feature_importance.csv` → `prune_features.py` escriu `csvfiles/features_pruned.txt` per a la propera execució.

Sense els fitxers obligatoris, el model no pot entrenar i l’app mostrarà un error.

---

## 📊 Mètriques i “Confiança Ajustada”

- **Accuracy del model**  
  És el percentatge de prediccions correctes (Over vs Under 2.5) en el conjunt de **test** (temporada 2024 o 20% cronològic). Es mostra a la secció “Veure Mètriques de Rendiment del Model”.

- **Confiança ajustada**  
  La probabilitat que es mostra (el **velocímetre** i el **Verdict**) és la sortida directa del model (P(Over 2.5)). Es pot interpretar com a “confiança del model” en que el partit superi els 2.5 gols.  
  En un context de confiança ajustada per l’accuracy:
  - Si l’accuracy del model és ~55–60%, les probabilitats no s’han de llegir com a probabilitats “reals” calibrades, sinó com a **nivell relatiu de confiança** (més % = el model creu més en Over 2.5).
  - Una “confiança ajustada” seria, conceptualment, no prendre el % al peu de la lletra i tenir en compte que el model encerta en ~55–60% dels casos; la UI deixa veure l’accuracy actual perquè l’usuari pugui contextualitzar les xifres.

La **matriu de confusió** (Under 2.5 vs Over 2.5) permet veure on falla més el model (falsos Over, falsos Under) i el **Top 10 Feature Importance** mostra quines variables pesen més (mv_ratio, H2H, dies de descans, etc.).

---

## 🗺️ Full de ruta (roadmap)

| Prioritat | Millora | Descripció |
|-----------|--------|------------|
| ~~Alta~~ | **xG (Expected Goals)** | Integrat via StatsBomb (tactical_danger_index) i resum per equip/temporada. |
| Alta | **Notificacions per baixes d’última hora** | Sistema (email / Telegram / in-app) que avisi quan l’API detecti baixes rellevants per a un partit guardat. |
| Mitjana | **Deep Learning per a l’empat** | Explorar un model (RNN/LSTM o similar) per predir resultat 1-X-2 o empat, complementant el model actual Over 2.5. |
| Mitjana | **Publicació a Streamlit Cloud** | Desplegar l’app a Streamlit Cloud amb Secrets per la clau API i, si cal, dades de mostra o enllaç a CSVs. |
| Baixa | **Més lligues i temporades** | Ampliar els CSVs a més competicions i anys per millorar cobertura i robustesa del model. |

---

## 📁 Estructura del projecte (resum)

```
ezmoney/
├── README.md
├── requirements.txt
├── run_oracle_pipeline.py       # Pipeline mestre: ESPN → StatsBomb → multi-league → model → poda
├── app.py                       # Dashboard Streamlit (Gauge, H2H, heatmap, radar)
├── football_pro_model.py       # Model Over 2.5, ratxes Home/Away, advantage, Poisson, ajust en viu
├── integrate_espn_data.py      # Micro-stats ESPN → espn_2025_summary.csv
├── process_statsbomb.py        # Event data / xG StatsBomb → statsbomb_summary.csv
├── integrate_multi_league_2026.py  # Dades lligues (córners, targetes, market_expectation)
├── prune_features.py           # Poda de variables (feature_importance → features_pruned.txt)
├── api_football.py             # Integració Football-Data.org (partits, alineacions, squad)
├── csvfiles/                   # (o csv_files/)
│   ├── clubs.csv
│   ├── club_games.csv
│   ├── games.csv
│   ├── players.csv
│   ├── espn_2025_summary.csv   # Opcional: resum ESPN per equip
│   ├── statsbomb_summary.csv   # Opcional: xG / tactical_danger per equip
│   ├── feature_importance.csv # Generat per football_pro_model
│   └── features_pruned.txt    # Generat per prune_features (llista de variables a conservar)
└── .streamlit/
    └── secrets.toml            # Opcional: api_football_key
```

---

## 📜 Llicència i ús

Projecte d’ús educatiu i personal. Les dades de Transfermarkt i Football-Data.org estan subjectes als seus respectius termes d’ús. No s’aconsella usar les prediccions com a única base per a decisions de joc o apostes.

---

*Football Oracle Pro — Model híbrid CSV + API amb ratxes Home/Away, ESPN, StatsBomb, multi-lliga, H2H, Poisson i ajust en viu.*
