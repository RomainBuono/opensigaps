# OpenSIGAPS — ValoMetric 5.0

Outil d'estimation de la valorisation SIGAPS MERRI 2022.
Moteur PubMed + NLP sémantique (multilingual-e5-small).

---

## ⚠️ Note sur le référentiel SIGAPS

Le fichier `sigaps_ref.csv` contient les rangs officiels des journaux SIGAPS.
Il n'est **pas inclus dans ce dépôt** pour des raisons liées aux droits sur les données.

L'image Docker locale, elle, **l'embarque** — voir la section Build ci-dessous.

---

## Structure du projet

```
OpenSIGAPS/
├── app.py                        ← Interface Streamlit
├── backend.py                    ← Moteur SIGAPS / PubMed / NLP
├── main.py                       ← Point d'entrée alternatif
├── clean_sigaps.py               ← Script nettoyage CSV
├── pyproject.toml                ← Dépendances (uv)
├── uv.lock                       ← Lockfile uv
├── Dockerfile                    ← Image Docker
├── docker-compose.yml            ← Orchestration
├── .dockerignore
├── .python-version
└── .streamlit/
    └── config.toml               ← Thème Streamlit

# Fichiers locaux uniquement (non versionnés) :
├── sigaps_ref.csv                ← Référentiel journaux (privé)
├── sigaps_ref.emb.npy            ← Cache embeddings (généré automatiquement)
└── multilingual-e5-small-onnx/  ← Modèle NLP local
```

---

## Prérequis

- **Docker Desktop** installé et démarré
- Le fichier `sigaps_ref.csv` disponible localement
- ~4 Go d'espace disque libre, 3 Go de RAM

Vérifier :
```bash
docker --version        # 24+
docker compose version  # 2.x
```

---

## Build local (avec CSV embarqué)

> Le CSV étant privé, le build se fait **uniquement en local**.
> L'image produite est autonome et n'a besoin d'aucune connexion pour tourner.

```bash
# 1. Se placer dans le dossier du projet
cd OpenSIGAPS/

# 2. Vérifier que le CSV est présent
ls sigaps_ref.csv

# 3. Builder l'image
#    → télécharge le modèle NLP (~500 Mo) + installe les dépendances
#    → première fois : 15-20 min / rebuilds suivants : ~2 min (cache Docker)
docker build -t opensigaps:5.0 .

# 4. Vérifier que l'image est créée
docker images | grep opensigaps
```

---

## Lancer l'application

```bash
# Démarrer
docker compose up -d

# Accéder à l'application
open http://localhost:8501       # Mac
# ou ouvrir http://localhost:8501 dans le navigateur

# Voir les logs
docker compose logs -f

# Arrêter
docker compose down
```

---

## Figer une version définitive (snapshot)

Pour archiver l'état exact de l'application avec ses données :

```bash
# Builder avec un tag daté
docker build -t opensigaps:5.0-$(date +%Y%m%d) .

# Exporter en archive autonome
docker save opensigaps:5.0-$(date +%Y%m%d) | gzip > opensigaps_snapshot_$(date +%Y%m%d).tar.gz
```

Cette archive `.tar.gz` contient **tout** : code, CSV, modèle NLP, dépendances.
Rechargeable dans 5 ans sans aucun accès réseau :

```bash
docker load < opensigaps_snapshot_20250220.tar.gz
docker run -p 8501:8501 opensigaps:5.0-20250220
```

---

## Mettre à jour le code sans rebuilder le modèle

Grâce au cache Docker multi-étapes, modifier `app.py` ou `backend.py` ne
retélécharge pas le modèle NLP :

```bash
# Modifier le code, puis :
docker build -t opensigaps:5.0 .   # ~2 min (le modèle est en cache)
docker compose up -d --force-recreate
```

---

## Commandes utiles

```bash
# État du conteneur
docker compose ps

# Entrer dans le conteneur (debug)
docker exec -it opensigaps bash

# Consommation mémoire
docker stats opensigaps

# Supprimer l'image et repartir de zéro
docker compose down --rmi all
docker system prune -f
```

---

## Résolution des problèmes courants

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `Port 8501 already in use` | Port occupé | `docker compose down` ou changer le port dans `docker-compose.yml` |
| `sigaps_ref.csv: not found` | CSV manquant au build | Vérifier qu'il est bien dans le dossier avant `docker build` |
| Modèle NLP non chargé | Réseau coupé pendant le build | `docker build --no-cache .` |
| Erreur mémoire au démarrage | RAM insuffisante | Allouer 3+ Go dans Docker Desktop → Préférences → Resources |
| Build très lent (1ère fois) | Téléchargement modèle 500 Mo | Normal — les rebuilds suivants sont rapides |
