---
title: OpenSIGAPS
emoji: 🧬
colorFrom: blue
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
---

# OpenSIGAPS

> Estimation automatisée de la valorisation SIGAPS · Algorithme MERRI 2022 · Aide au choix de journal propulsée par IA

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![PubMed](https://img.shields.io/badge/PubMed-API_Entrez-326599?style=flat-square&logo=ncbi&logoColor=white)
![Scopus](https://img.shields.io/badge/Scopus-Elsevier-FF6D00?style=flat-square&logo=elsevier&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-multilingual--e5--small-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## À propos

**OpenSIGAPS** est une application web permettant aux chercheurs et médecins hospitaliers d'estimer leur valorisation SIGAPS selon l'algorithme officiel MERRI 2022, sans avoir à saisir manuellement leurs publications dans la plateforme nationale.

Elle interroge simultanément **PubMed (API Entrez) et Scopus (API Elsevier)** via une recherche fédérée pour récupérer automatiquement les publications. L'application calcule les scores de présence et fractionnaires, et propose un moteur sémantique d'aide au choix de journal. Ce moteur intègre une **détection hiérarchique du domaine clinique** et un système de recommandation basé sur **multilingual-e5-small** (Microsoft).

---

## 🚀 Nouvelles Fonctionnalités 

**🧬 Analyse SIGAPS & Recherche Fédérée**
- Recherche automatique simultanée sur PubMed et Scopus.
- Déduplication intelligente cross-source (par DOI et similarité de titre).
- Calcul des scores Présence (C1 × C2) et Fractionnaire (MERRI 2022).
- Mode individuel et mode équipe / service (application de la règle de la meilleure position).
- Export Excel complet avec récapitulatif financier.

**🔎 Aide au choix de journal (Pipeline NLP)**
- **Inférence de domaine hiérarchique :** Détection automatique de la spécialité (ex: Oncologie Thoracique, Hématologie, Epidémiologie).
- **Boost Thématique :** Surpondération mathématique des journaux dont la ligne éditoriale correspond au domaine clinique détecté.
- Score de pertinence basé sur la similarité cosinus (embeddings 384 dims, ONNX int8).
- Suggestion de publications similaires issues de PubMed et Scopus.
- Filtrage temps réel par rang SIGAPS, IF, et indexation Medline.

---

## 🏗️ Architecture Technique

L'application respecte les principes de la **Clean Architecture**, séparant l'interface, la logique métier et l'accès aux données.

```text
 PubMed (Entrez) & Scopus API
               │
               ▼
         src/fetchers/    ──── Recherche Fédérée & Déduplication (Smart Fetching)
               │
            src/nlp/      ──── 1. Inférence de Domaine Hiérarchique (Classification)
               │               2. Extraction intelligente de mots-clés spécifiques
               │               3. Encodage dense : multilingual-e5-small
               │               4. Semantic Ranker (Cosinus + Boost Thématique)
               ▼
            app.py        ──── Interface Streamlit Sécurisée
```

---

## 📂 Structure du projet

```text
OpenSIGAPS/
├── Dockerfile                ← Recette Docker optimisée (Multi-stage, CPU-only, User non-root)
├── .dockerignore             ← Bouclier de build
├── .env.example              ← Template des clés API (à copier en .env)
├── pyproject.toml & uv.lock  ← Dépendances strictes (gestion ultra-rapide via 'uv')
├── app.py                    ← Point d'entrée de l'interface Streamlit
├── anonymize.py              ← Script de génération du dataset de démo
│
├── src/                      ← Cœur métier modulaire
│   ├── core/                 ← Entités (Article, ProcessedQuery, SuggestedJournal)
│   ├── fetchers/             ← Connecteurs API (PubMedFetcher, ScopusFetcher)
│   ├── nlp/                  ← Moteurs IA (QueryProcessor, DomainInference, SemanticRanker)
│   ├── repositories/         ← Accès base de données (SigapsRefDB)
│   └── services/             ← Orchestrateurs (JournalRecommendationService)
│
└── data/
    └── processed/
        └── sigaps_demo.csv   ← Dataset de démo anonymisé (Inclus dans le dépôt)
```

---

## ⚙️ Installation et Déploiement

### Option A : Déploiement Production (Docker - Recommandé)

L'application est dockerisée de manière stricte pour les environnements hospitaliers (image allégée sans drivers GPU inutiles, exécution sans privilèges `root`).

1. **Cloner le dépôt et configurer l'environnement :**
   ```bash
   git clone [https://github.com/RomainBuono/opensigaps.git](https://github.com/RomainBuono/opensigaps.git)
   cd opensigaps
   
   # Créer le fichier des secrets à partir de l'exemple
   cp .env.example .env
   # Éditer .env avec vos clés API (NCBI_API_KEY, SCOPUS_API_KEY)
   ```

2. **Ajouter le référentiel SIGAPS :**
   Placez votre fichier officiel `sigaps_ref.csv` dans le dossier `data/processed/`. S'il est absent, l'application basculera automatiquement sur le fichier de démo.

3. **Compiler et lancer l'image :**
   ```bash
   # Construction ultra-rapide via uv
   docker build -t opensigaps:latest .
   
   # Lancement avec injection sécurisée des variables d'environnement
   docker run -d --name sigaps_app -p 8501:8501 --env-file .env opensigaps:latest
   ```
   L'application est accessible sur `http://localhost:8501`.

### Option B : Développement Local

Prérequis : Python 3.11+ et [`uv`](https://github.com/astral-sh/uv).

```bash
# Installer les dépendances en respectant le lockfile
uv sync --frozen

# Lancer l'application
uv run streamlit run app.py
```

### Option C : Mode Démo (Données Anonymisées)

Si vous souhaitez exposer ou faire tester l'application publiquement sans révéler la matrice SIGAPS confidentielle de votre établissement, un script d'anonymisation est fourni.

Ce script crée un fichier `sigaps_demo.csv` qui conserve les vrais noms des journaux scientifiques (indispensables au bon fonctionnement du moteur d'Intelligence Artificielle), mais **mélange aléatoirement les Rangs SIGAPS et les Impact Factors**.

```bash
# Générer un nouveau fichier de démo à partir de votre référentiel
uv run anonymize.py
```

Si le fichier secret `sigaps_ref.csv` n'est pas détecté au lancement, l'application chargera automatiquement ce fichier de démonstration. L'expérience utilisateur (UX) et le moteur de suggestion sémantique resteront 100% fonctionnels, mais les calculs financiers générés seront fictifs.

---

## 🔒 Note sur les Données et Référentiels

- **Fichier SIGAPS officiel (`sigaps_ref.csv`) :** Ce fichier n'est **pas distribué** dans ce dépôt pour des raisons de confidentialité institutionnelle. Seul le dataset de démo y figure.
- **Modèles d'IA :** Les modèles de langage (HuggingFace) sont instanciés en mode *Air-Gapped* (intégrés directement dans l'image ou mis en cache localement) pour respecter les contraintes des pare-feux hospitaliers.

**Avertissement :** OpenSIGAPS est un outil d'estimation développé pour l'aide à la décision. La valorisation officielle et légale reste celle validée par la plateforme nationale [sigaps.fr](https://www.sigaps.fr).

---

## 📚 Références

- DRCI AP-HP. *Modifications SIGAPS MERRI 2022*. [PDF officiel](https://recherche-innovation.aphp.fr/wp-content/blogs.dir/77/files/2022/04/SIGAPS-Nouveautes-diffusion-V2.pdf)
- Lepage E. *et al.* (2015). SIGAPS score and medical publication evaluation system. *IRBM*, 36(2).
- Wang L. *et al.* (2024). Multilingual E5 Text Embeddings. *Microsoft Research*.

---

## 👨‍💻 Auteur

Architecturé et développé par **Romain Buono** · [@RomainBuono](https://github.com/RomainBuono)
