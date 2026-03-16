# Fichier : src/core/entities.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass(frozen=True)
class Article:
    """
    Représente un article scientifique agnostique (indépendant de sa source).
    Immuable (frozen=True) pour garantir l'intégrité des données dans le pipeline.
    """
    id: str                  # Identifiant unique (ex: "PMID:12345" ou "SCOPUS:6789")
    source: str              # Nom du fetcher (ex: "PubMed", "Scopus")
    title: str               # Titre brut de l'article
    journal_name: str        # Nom complet de la revue
    publication_year: int    # Année de publication
    doi: str = ""            # Digital Object Identifier (clé primaire pour dédoublonnage)
    pmid: str = ""           # PubMed ID (optionnel si source = Scopus)
    issn: str = ""           # International Standard Serial Number
    authors_list: List[str] = field(default_factory=list) # Liste des auteurs


@dataclass
class ProcessedQuery:
    """
    Résultat de l'analyse NLP du titre soumis par le chercheur.
    Contient tout ce dont les API et le moteur de ranking ont besoin.
    """
    original_title: str
    search_keywords: List[str]  # Mots-clés extraits pour la requête API (PubMed/Scopus)
    
    # Le vecteur E5 (optionnel au cas où le modèle E5 ne charge pas)
    # eq=False évite les erreurs lors de la comparaison de dataclasses contenant des arrays numpy
    embedding_vector: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    
    # Métadonnées additionnelles (ex: domaine inféré, niveau de confiance)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestedJournal:
    """
    Résultat final agrégé renvoyé à l'interface utilisateur (Streamlit).
    """
    nlm_id: str                 # Identifiant unique du journal dans le référentiel SIGAPS
    journal_name: str           # Nom complet
    medline_ta: str             # Abréviation Medline
    similarity_score: float     # Score normalisé [0.0, 1.0] issu du SemanticRanker
    matched_articles_count: int # Nombre d'articles similaires trouvés dans ce journal
    
    # Données SIGAPS
    rank_sigaps: str = "NC"     # Rang (A+, A, B, C, D, E, NC)
    rank_source: str = "heuristique" # "csv" ou "heuristique"
    impact_factor: str = ""     # Ex: "12.5"
    medline_indexed: str = ""   # Statut d'indexation
    
    # Preuves (pour l'expander Streamlit)
    example_titles: List[Dict[str, str]] = field(default_factory=list) # [{"title": "...", "pmid": "..."}]