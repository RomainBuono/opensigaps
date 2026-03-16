# Fichier : src/fetchers/scopus_fetcher.py

import os
import requests
from typing import List

from src.core.entities import Article
from src.core.interfaces import ArticleFetcher

class ScopusFetcher(ArticleFetcher):
    """
    Connecteur pour l'API Elsevier Scopus.
    """
    
    BASE_URL = "https://api.elsevier.com/content/search/scopus"

    def __init__(self, api_key: str = None):
        """Une clé API Scopus est strictement obligatoire."""
        self.api_key = api_key or os.getenv("SCOPUS_API_KEY")

    @property
    def source_name(self) -> str:
        return "Scopus"

    def fetch_by_keywords(self, keywords: List[str], max_results: int = 100) -> List[Article]:
        if not keywords or not self.api_key:
            return []

        query_parts = [f'"{kw}"' for kw in keywords]
        
        queries_to_try = []
        for limit in [len(query_parts), 3, 2, 1]:
            q = "TITLE-ABS-KEY(" + " AND ".join(query_parts[:limit]) + ")"
            if q not in queries_to_try:
                queries_to_try.append(q)

        headers = {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}

        for scopus_query in queries_to_try:
            params = {
                "query": scopus_query,
                "count": str(max_results),
                "field": "dc:title,prism:publicationName,prism:coverDate,prism:doi,prism:issn,dc:identifier" 
            }
            try:
                response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Si on trouve des articles, on les parse et on stoppe la boucle
                if data.get("search-results", {}).get("entry", []):
                    return self._parse_json_to_articles(data)
            except Exception:
                continue
                
        return []

    def _parse_json_to_articles(self, data: dict) -> List[Article]:
        """Transforme le payload JSON d'Elsevier en objets métiers locaux."""
        articles = []
        entries = data.get("search-results", {}).get("entry", [])
        
        for entry in entries:
            try:
                title = entry.get("dc:title", "")
                journal_name = entry.get("prism:publicationName", "")
                
                # Scopus renvoie la date sous format "YYYY-MM-DD"
                date_str = entry.get("prism:coverDate", "")
                year = int(date_str[:4]) if date_str and len(date_str) >= 4 else 0
                
                doi = entry.get("prism:doi", "")
                issn = entry.get("prism:issn", "")
                
                # Scopus ID ressemble à "SCOPUS_ID:85123..."
                raw_id = entry.get("dc:identifier", "")
                scopus_id = raw_id.replace("SCOPUS_ID:", "") if raw_id else ""

                if title and journal_name:
                    articles.append(Article(
                        id=f"SCOPUS:{scopus_id}" if scopus_id else f"DOI:{doi}",
                        source=self.source_name,
                        title=title,
                        journal_name=journal_name,
                        publication_year=year,
                        doi=doi,
                        issn=issn
                        # Scopus ne fournit pas le PMID dans cette vue de base
                    ))
            except Exception:
                continue
                
        return articles