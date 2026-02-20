<div align="center">

# ⚕️ OpenSIGAPS — 1.0

**Estimation de valorisation SIGAPS · Algorithme MERRI 2022 · Aide au choix de journal**

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

![OpenSIGAPS Banner](https://img.shields.io/badge/SIGAPS-MERRI%202022-1e3a5f?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQTEwIDEwIDAgMCAwIDIgMTJhMTAgMTAgMCAwIDAgMTAgMTAgMTAgMTAgMCAwIDAgMTAtMTBBMTAgMTAgMCAwIDAgMTIgMm0tMSAxNXYtNEg3bDUtOXY0aDRsLTUgOXoiLz48L3N2Zz4=)

</div>

---

## À quoi ça sert ?

Les publications scientifiques des médecins hospitaliers sont valorisées financièrement par l'État via le système **SIGAPS** (Système d'Interrogation, de Gestion et d'Analyse des Publications Scientifiques), qui conditionne une part significative des dotations **MERRI** (~61% de la dotation socle).

**OpenSIGAPS** automatise ce qui prend habituellement des heures :

- ✅ Récupérer ses publications depuis **PubMed** en quelques secondes
- ✅ Calculer ses **scores SIGAPS** (présence, fractionnaire MERRI 2022)
- ✅ Estimer la **valeur financière annuelle** de sa production
- ✅ Identifier les **meilleurs journaux cibles** pour ses prochaines soumissions

---

## Fonctionnalités

### 🧬 Analyse SIGAPS
- Recherche automatique sur **PubMed** par nom d'auteur
- Déduplication et validation manuelle des publications
- Calcul du **score présence** (C1 × C2) et du **score fractionnaire** (MERRI 2022)
- Modes **individuel** et **équipe/service**
- Export **Excel** complet avec synthèse financière

### 🔎 Aide au choix de journal
- Suggestion de journaux par **similarité sémantique NLP** à partir du titre de l'article
- Pipeline en 5 niveaux de relaxation sur PubMed (jusqu'à 100 articles analysés)
- Tri par **rang SIGAPS** (A+ → NC) et **Impact Factor**
- Recherche par **nom de journal** ou **DOI**
- Filtres : rang, IF minimum, indexation Medline
- Export Excel des suggestions · Liens PubMed directs

### 📖 Méthodologie
- Documentation complète de l'algorithme MERRI 2022
- Coefficients C1/C2, matrice des scores, exemples numériques
- Description du moteur NLP et du calcul du score de pertinence

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit 1.42 |
| NLP — moteur principal | multilingual-e5-small (ONNX int8) |
| NLP — fallback | SentenceTransformers / Jaccard trigrammes |
| Inférence | ONNX Runtime (~8 ms/phrase) |
| API données | PubMed Entrez / NLM Catalog |
| Visualisation | Plotly |
| Export | openpyxl (Excel) |
| Déploiement | Docker multi-stage |

---

## Démarrage rapide

### Avec Docker (recommandé)

```bash
git clone https://github.com/RomainBuono/opensigaps.git
cd opensigaps

# Ajouter votre référentiel SIGAPS
cp /chemin/vers/sigaps_ref.csv .

# Build + lancement (première fois ~15 min)
docker compose up -d

# → http://localhost:8501
```

### En local (développement)

```bash
git clone https://github.com/RomainBuono/opensigaps.git
cd opensigaps

# Avec uv (recommandé)
uv sync
uv run streamlit run app.py

# Ou avec pip classique
pip install -r requirements.txt
streamlit run app.py
```

---

## Structure du projet

```
opensigaps/
├── app.py                 ← Interface Streamlit (2 400 lignes)
├── backend.py             ← Moteur SIGAPS / PubMed / NLP (1 550 lignes)
├── main.py                ← Point d'entrée alternatif
├── clean_sigaps.py        ← Utilitaire nettoyage CSV
├── .streamlit/
│   └── config.toml        ← Thème light (navy + Playfair Display)
├── Dockerfile             ← Build multi-stage avec modèle ONNX embarqué
├── docker-compose.yml
└── requirements.txt       ← Dépendances figées
```

---

## Algorithme SIGAPS MERRI 2022

```
Score Présence    = C1 (position) × C2 (rang revue)
Score Fractionnaire = (C1_auteur / Σ C1_tous_auteurs) × C2
```

| Position | C1 | | Rang | C2 |
|----------|----|-|------|-----|
| 1er / Dernier | 4 | | A+ | 14 |
| 2ème | 3 | | A | 8 |
| 3ème | 2 | | B | 6 |
| ADA (≥6 auteurs) | 3 | | C | 4 |
| Autre | 1 | | D | 3 |
| | | | E | 2 |
| | | | NC | 1 |

> Score maximum : **4 × 14 = 56 pts** (1er auteur, revue A+)  
> Réforme 2021 : ajout rang A+, position 3ème, position ADA, score fractionnaire

---

## Avertissement

> OpenSIGAPS est un **outil d'estimation**. La valorisation officielle est effectuée par la plateforme nationale [SIGAPS](https://www.sigaps.fr) et validée par la DRCI de votre établissement. Les rangs affichés sont des approximations à valider avec votre référentiel institutionnel.

---

## Références

- DRCI AP-HP. *Modifications SIGAPS MERRI 2022*. [PDF officiel](https://recherche-innovation.aphp.fr/wp-content/blogs.dir/77/files/2022/04/SIGAPS-Nouveautes-diffusion-V2.pdf)
- Lepage E. *et al.* (2015). SIGAPS score and medical publication evaluation system. *IRBM*, 36(2). [DOI](https://doi.org/10.1016/j.irbm.2015.01.006)
- Wang L. *et al.* (2024). Multilingual E5 Text Embeddings. *Microsoft Research*. [arXiv:2402.05672](https://arxiv.org/abs/2402.05672)

---

<div align="center">

Fait avec ☕ et beaucoup de PubMed API calls

</div>

