# Fichier : src/core/__init__.py

from .entities import Article, ProcessedQuery, SuggestedJournal
from .interfaces import ArticleFetcher, QueryProcessor, SemanticRanker

__all__ = [
    "Article",
    "ProcessedQuery",
    "SuggestedJournal",
    "ArticleFetcher",
    "QueryProcessor",
    "SemanticRanker",
]