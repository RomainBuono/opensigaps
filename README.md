# 🏥 OpenSIGAPS — 1.0

> Estimation automatisée de la valorisation SIGAPS · Algorithme MERRI 2022 · Aide au choix de journal

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PubMed](https://img.shields.io/badge/PubMed-API_Entrez-326599?style=flat-square&logo=ncbi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-multilingual--e5--small-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-int8-005CED?style=flat-square&logo=onnx&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white)

---

## À propos

**OpenSIGAPS** est une application web permettant aux chercheurs et médecins hospitaliers d'estimer leur valorisation SIGAPS selon l'algorithme officiel MERRI 2022, sans avoir à saisir manuellement leurs publications dans la plateforme nationale.

Elle interroge **PubMed via l'API Entrez** pour récupérer automatiquement les publications, calcule les scores de présence et fractionnaires, et propose un moteur sémantique d'aide au choix de journal basé sur **multilingual-e5-small** (Microsoft).

---

## Fonctionnalités

**🧬 Analyse SIGAPS**
- Recherche automatique des publications sur PubMed par nom d'auteur
- Calcul des scores Présence (C1 × C2) et Fractionnaire (MERRI 2022)
- Mode individuel et mode équipe / service
- Revue et correction manuelle des publications (position, rang)
- Export Excel complet avec récapitulatif financier

**🔎 Aide au choix de journal**
- Recherche par titre de journal ou DOI
- Suggestion sémantique par titre d'article (cascade PubMed + NLP)
- Score de pertinence basé sur la similarité cosinus (embeddings 384 dims)
- Filtrage par rang SIGAPS, IF, indexation Medline
- Tri interactif (rang, IF, pertinence)
- Évolution historique du rang et de l'IF (graphique Plotly)
- Export Excel des suggestions

**📖 Méthodologie**
- Coefficients C1 / C2 officiels MERRI 2022
- Matrice des scores de présence
- Documentation de l'algorithme NLP

---

## Architecture technique

```
PubMed API (Entrez)
       │
       ▼
  backend.py ──── Cascade de recherche (5 niveaux)
       │           Scoring SIGAPS (C1 × C2)
       │           NLP : multilingual-e5-small (ONNX int8)
       │                 └─ fallback : PyTorch → Jaccard trigrammes
       ▼
   app.py ──────── Interface Streamlit
                   Thème médical-éditorial (Playfair Display + Sora)
                   Visualisations Plotly
```

**Pipeline NLP — suggestion de journal :**
1. Extraction des mots-clés du titre (stopwords médicaux exclus)
2. Cascade de relaxation PubMed (AND → OR → terme seul) jusqu'à ≥ 15 résultats
3. Encodage ONNX int8 (~8 ms / phrase vs ~30 ms PyTorch)
4. Similarité cosinus → agrégation par journal → score de pertinence normalisé
5. Classement par rang SIGAPS puis IF décroissant

---

## Structure du projet

```
OpenSIGAPS/
├── app.py                        ← Interface Streamlit (2 400 lignes)
├── backend.py                    ← Moteur SIGAPS / PubMed / NLP (1 550 lignes)
├── main.py                       ← Point d'entrée alternatif
├── clean_sigaps.py               ← Script nettoyage du référentiel CSV
├── pyproject.toml                ← Dépendances gérées par uv
├── uv.lock                       ← Lockfile figé
├── .python-version               ← Python 3.11
└── .streamlit/
    └── config.toml               ← Thème clair (couleurs, police)

# Fichiers locaux uniquement — non versionnés (.gitignore) :
├── sigaps_ref.csv                ← Référentiel journaux SIGAPS (privé)
├── sigaps_ref.emb.npy            ← Cache embeddings NumPy (auto-généré)
├── sigaps_ref.emb.ids            ← Index NLM_IDs du cache (auto-généré)
└── multilingual-e5-small-onnx/  ← Modèle NLP téléchargé localement
```

---

## Installation et lancement

### Prérequis

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) — gestionnaire de paquets rapide

```bash
# Installer uv (si besoin)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Lancer l'application

```bash
# Cloner le dépôt
git clone https://github.com/RomainBuono/opensigaps.git
cd opensigaps/

# Installer les dépendances
uv sync

# Lancer
uv run streamlit run app.py
# → http://localhost:8501
```

### Référentiel SIGAPS

L'application fonctionne sans `sigaps_ref.csv` (les rangs sont alors estimés par heuristique NLM).
Pour des résultats précis, placez votre fichier `sigaps_ref.csv` à la racine du projet.

Colonnes minimales attendues :

| Colonne | Description |
|---------|-------------|
| `NLM_ID` | Identifiant NLM unique du journal |
| `Journal` | Nom complet du journal |
| `Latest_Rank` | Rang SIGAPS (`A+`, `A`, `B`, `C`, `D`, `E`, `NC`) |
| `YYYY_IF` *(optionnel)* | Impact Factor par année (ex: `2022_IF`) |
| `Indexation_Medline` *(optionnel)* | `Oui` / `Non` |

---

## ⚠️ Note juridique

Le fichier `sigaps_ref.csv` n'est **pas distribué** dans ce dépôt.
Les données de rangs SIGAPS sont produites et maintenues institutionnellement.
L'application est un **outil d'estimation** — la valorisation officielle reste celle de la plateforme [sigaps.fr](https://www.sigaps.fr).

---

## Références

- DRCI AP-HP. *Modifications SIGAPS MERRI 2022*. [PDF officiel](https://recherche-innovation.aphp.fr/wp-content/blogs.dir/77/files/2022/04/SIGAPS-Nouveautes-diffusion-V2.pdf)
- Lepage E. *et al.* (2015). SIGAPS score and medical publication evaluation system. *IRBM*, 36(2). [DOI](https://doi.org/10.1016/j.irbm.2015.01.006)
- Wang L. *et al.* (2024). Multilingual E5 Text Embeddings. *Microsoft Research*. [arXiv:2402.05672](https://arxiv.org/abs/2402.05672)

---

## Auteur

Développé par **Romain Buono** · [@RomainBuono](https://github.com/RomainBuono)
