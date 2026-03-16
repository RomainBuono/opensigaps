# Fichier : src/services/recommendation_service.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set

from src.core.entities import Article, SuggestedJournal
from src.core.interfaces import QueryProcessor, ArticleFetcher, SemanticRanker

# Configuration d'un logger basique pour le monitoring
logger = logging.getLogger(__name__)

class JournalRecommendationService:
    """
    Service d'orchestration de haut niveau.
    Il coordonne l'analyse NLP, la récupération concurrente des articles,
    le dédoublonnage, et le ranking sémantique.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        fetchers: List[ArticleFetcher],
        semantic_ranker: SemanticRanker,
        max_results_per_fetcher: int = 100
    ):
        """Injection des dépendances (les contrats)."""
        self._processor = query_processor
        self._fetchers = fetchers
        self._ranker = semantic_ranker
        self._max_results = max_results_per_fetcher

    def get_suggestions(self, article_title: str) -> List[SuggestedJournal]:
        """
        Point d'entrée principal appelé par l'interface Streamlit.
        """
        if not article_title or len(article_title.strip()) < 10:
            logger.warning("Titre trop court ou invalide soumis.")
            return []

        # 1. Préparation de la requête (NLP local)
        logger.info(f"Traitement du titre : {article_title}")
        query = self._processor.process(article_title)
        
        if not query.search_keywords:
            logger.warning("Aucun mot-clé pertinent n'a pu être extrait.")
            return []

        # 2. Récupération concurrente (Scatter)
        logger.info(f"Recherche API avec les mots-clés : {query.search_keywords}")
        raw_articles = self._fetch_all_concurrently(query.search_keywords)

        if not raw_articles:
            logger.info("Aucun article trouvé dans les bases de données.")
            return []

        # 3. Dédoublonnage (Gather & Clean)
        unique_articles = self._deduplicate(raw_articles)
        logger.info(f"Articles récupérés : {len(raw_articles)} -> Dédoublonnés : {len(unique_articles)}")

        # 4. Reranking et Agrégation
        suggestions = self._ranker.rank_and_aggregate(
            query=query, 
            articles=unique_articles,
            max_journals=10
        )

        return suggestions

    def _fetch_all_concurrently(self, keywords: List[str]) -> List[Article]:
        """
        Exécute tous les fetchers (PubMed, Scopus...) en parallèle via des Threads.
        Divise le temps d'attente total par le nombre de fetchers.
        """
        all_articles: List[Article] = []
        
        # Le ThreadPoolExecutor gère un pool de threads (1 thread par fetcher)
        with ThreadPoolExecutor(max_workers=len(self._fetchers)) as executor:
            # On lance les tâches de manière asynchrone
            future_to_fetcher = {
                executor.submit(fetcher.fetch_by_keywords, keywords, self._max_results): fetcher 
                for fetcher in self._fetchers
            }
            
            # On récolte les résultats au fur et à mesure qu'ils se terminent
            for future in as_completed(future_to_fetcher):
                fetcher = future_to_fetcher[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Le fetcher {fetcher.source_name} a généré une erreur : {e}")
                    
        return all_articles

    def _deduplicate(self, articles: List[Article]) -> List[Article]:
        """
        Fusionne les articles identiques provenant de sources différentes.
        Priorité 1 : Le DOI (Digital Object Identifier) - 100% fiable.
        Priorité 2 : Le Titre (normalisé en minuscules) - Heuristique.
        """
        unique_articles: List[Article] = []
        seen_dois: Set[str] = set()
        seen_titles: Set[str] = set()

        for article in articles:
            # 1. Vérification par DOI
            if article.doi:
                clean_doi = article.doi.strip().lower()
                if clean_doi in seen_dois:
                    continue  # Déjà vu, on ignore
                seen_dois.add(clean_doi)
            
            # 2. Vérification par Titre exact (pour les articles sans DOI)
            clean_title = article.title.strip().lower()
            if clean_title in seen_titles:
                continue # Déjà vu, on ignore
            seen_titles.add(clean_title)

            # Si on arrive ici, l'article est unique
            unique_articles.append(article)

        return unique_articles