# Fichier : src/core/interfaces.py

import abc
from typing import List
from .entities import Article, ProcessedQuery, SuggestedJournal

class QueryProcessor(abc.ABC):
    """
    Contrat pour le moteur d'analyse de la requête initiale.
    """
    @abc.abstractmethod
    def process(self, title: str) -> ProcessedQuery:
        """
        Analyse le titre et génère les mots-clés et le vecteur d'embedding.
        
        Args:
            title (str): Le titre de l'article saisi par l'utilisateur.
            
        Returns:
            ProcessedQuery: L'objet contenant les tokens et le vecteur.
        """
        pass


class ArticleFetcher(abc.ABC):
    """
    Contrat pour les connecteurs aux bases de données bibliographiques.
    """
    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        """Nom de la source de données (ex: 'PubMed')."""
        pass

    @abc.abstractmethod
    def fetch_by_keywords(self, keywords: List[str], max_results: int = 100) -> List[Article]:
        """
        Récupère une liste d'articles correspondants aux mots-clés.
        
        Args:
            keywords (List[str]): Les mots-clés extraits de la requête.
            max_results (int): Limite théorique de résultats à ramener.
            
        Returns:
            List[Article]: Une liste d'objets Article standardisés.
        """
        pass


class SemanticRanker(abc.ABC):
    """
    Contrat pour le moteur de scoring sémantique et d'agrégation.
    """
    @abc.abstractmethod
    def rank_and_aggregate(
        self, 
        query: ProcessedQuery, 
        articles: List[Article], 
        max_journals: int = 10
    ) -> List[SuggestedJournal]:
        """
        Calcule la similarité entre la requête et les articles, puis agrège par journal.
        
        Args:
            query (ProcessedQuery): La requête contenant le vecteur d'embedding.
            articles (List[Article]): La liste dédoublonnée d'articles ramenés par les fetchers.
            max_journals (int): Nombre maximum de journaux à suggérer.
            
        Returns:
            List[SuggestedJournal]: La liste finale, triée par pertinence.
        """
        pass