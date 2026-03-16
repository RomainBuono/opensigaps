# Fichier : src/fetchers/pubmed_fetcher.py

import os
import requests
import xml.etree.ElementTree as ET
from typing import List

from src.core.entities import Article
from src.core.interfaces import ArticleFetcher

class PubMedFetcher(ArticleFetcher):
    """
    Connecteur pour l'API PubMed (NCBI E-utilities).
    Utilise esearch pour trouver les PMIDs, puis efetch pour récupérer le XML.
    """
    
    BASE_URL_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_URL_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, api_key: str = None):
        """
        L'API key est optionnelle mais recommandée (passe la limite de 3 à 10 requêtes/sec).
        Si non fournie, on tente de la lire depuis les variables d'environnement.
        """
        self.api_key = api_key or os.getenv("NCBI_API_KEY")

    @property
    def source_name(self) -> str:
        return "PubMed"

    def fetch_by_keywords(self, keywords: List[str], max_results: int = 100) -> List[Article]:
        if not keywords:
            return []

        # 1. Construction de la cascade (du plus strict au plus large)
        query_parts = [f'"{kw}"[Title/Abstract]' for kw in keywords]
        queries_to_try = []
        for limit in [len(query_parts), max(1, len(query_parts)-1), 3, 2, 1]:
            q = " AND ".join(query_parts[:limit])
            if q not in queries_to_try:
                queries_to_try.append(q)
                
        # 2. Le nouveau système d'accumulation (Le seau)
        all_pmids = []
        seen_pmids = set() # Pour éviter de compter les doublons
        min_target = 50    # 🚀 NOTRE NOUVELLE RÈGLE : On veut au moins 50 articles !

        for q in queries_to_try:
            found_pmids = self._get_pmids(q, max_results)
            
            # On verse les nouveaux poissons dans le seau
            for pid in found_pmids:
                if pid not in seen_pmids:
                    seen_pmids.add(pid)
                    all_pmids.append(pid)


            # 🚀 3. LA CONDITION INTELLIGENTE : A-t-on atteint notre quota minimum ?
            if len(all_pmids) >= min_target:
                break 

        # On s'assure de ne pas dépasser la limite absolue (ex: 100)
        final_pmids = all_pmids[:max_results]

        if not final_pmids:
            return []
            
        return self._fetch_details(final_pmids)

    def _get_pmids(self, query: str, max_results: int) -> List[str]:
        """Étape 1 : Interroge esearch pour obtenir une liste d'IDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max_results)
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(self.BASE_URL_SEARCH, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"⚠️ Erreur PubMed (esearch) : {e}")
            return []

    def _fetch_details(self, pmids: List[str]) -> List[Article]:
        """Étape 2 : Télécharge et parse le XML pour les PMIDs trouvés."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(self.BASE_URL_FETCH, params=params, timeout=15)
            response.raise_for_status()
            return self._parse_xml_to_articles(response.text)
        except Exception as e:
            print(f"⚠️ Erreur PubMed (efetch) : {e}")
            return []

    def _parse_xml_to_articles(self, xml_data: str) -> List[Article]:
        """Parse le XML complexe de PubMed de manière défensive."""
        articles = []
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return articles

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                pmid = article_elem.findtext(".//PMID", default="")
                title = article_elem.findtext(".//ArticleTitle", default="")
                journal_name = article_elem.findtext(".//Journal/Title", default="")
                issn = article_elem.findtext(".//Journal/ISSN", default="")
                
                # Extraction du NLM_ID pour la jointure SIGAPS
                nlm_tag = article_elem.find(".//MedlineJournalInfo/NlmUniqueID")
                nlm_id = (nlm_tag.text or "").strip() if nlm_tag is not None else ""
                
                # Extraction de l'année
                year_str = article_elem.findtext(".//JournalIssue/PubDate/Year")
                if not year_str: # Fallback sur MedlineDate
                    medline_date = article_elem.findtext(".//JournalIssue/PubDate/MedlineDate", default="0")
                    year_str = medline_date[:4] if medline_date else "0"
                year = int(year_str) if year_str.isdigit() else 0

                # Extraction du DOI
                doi = ""
                for el in article_elem.findall(".//ArticleId"):
                    if el.attrib.get("IdType") == "doi":
                        doi = el.text
                        break

                if title and journal_name:
                    # 1. On crée l'article de façon classique et 100% conforme au schéma
                    art = Article(
                        id=f"PMID:{pmid}",
                        source=self.source_name,
                        title=title,
                        journal_name=journal_name,
                        authors_list=[], 
                        publication_year=year,
                        doi=doi,
                        pmid=pmid,
                        issn=nlm_id  
                    )
                    articles.append(art)
            except Exception as e:
                continue
                
        return articles