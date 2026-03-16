# Fichier : src/fetchers/__init__.py

from .pubmed_fetcher import PubMedFetcher
from .scopus_fetcher import ScopusFetcher

__all__ = [
    "PubMedFetcher",
    "ScopusFetcher"
]