# Fichier : src/nlp/semantic_ranker.py

import numpy as np
from typing import List, Dict

from src.core.entities import Article, ProcessedQuery, SuggestedJournal
from src.core.interfaces import SemanticRanker

class CosineSemanticRanker(SemanticRanker):
    """
    Moteur de ranking utilisant la similarité cosinus sur des vecteurs denses
    et appliquant un boost thématique si applicable.
    """
    
    def __init__(self, embedder, sigaps_repo=None, domain_boost_factor: float = 0.40):
        """
        Args:
            embedder: Le modèle pour vectoriser les titres des articles candidats.
            sigaps_repo: Un objet ou dictionnaire simulant ta base 'sigaps_ref.csv'.
            domain_boost_factor: Le multiplicateur de score thématique.
        """
        self._embedder = embedder
        self._sigaps_repo = sigaps_repo
        self._domain_boost_factor = domain_boost_factor

    def rank_and_aggregate(
        self, 
        query: ProcessedQuery, 
        articles: List[Article], 
        max_journals: int = 50  
    ) -> List[SuggestedJournal]:
        
        # SÉCURITÉ ANTI-TRONCATURE : On force 50 au cas où le service parent enverrait 10
        max_journals = 50 

        if not articles or query.embedding_vector is None:
            return []

        # 1. Vectorisation Asymétrique des articles candidats (Batch)
        article_texts = []
        for a in articles:
            # On construit le super-vecteur : Titre pur + Concepts purs
            text = a.title
            # On utilise getattr pour éviter un crash si l'objet Article est une ancienne version en cache
            if getattr(a, "concepts", ""):
                text += f". {a.concepts}"
            article_texts.append(text)

        try:
            # Note : on retire le préfixe 'passage:' qui était spécifique à e5. bge-m3 n'en a pas besoin.
            art_vecs = self._embedder.encode(
                article_texts, 
                batch_size=32, 
                normalize_embeddings=True, 
                convert_to_numpy=True,
                show_progress_bar=False 
            )
        except Exception as e:
            logger.error(f"Erreur lors de la vectorisation NLP : {e}")
            return []

        # 2. Produit Scalaire
        sims = (art_vecs @ query.embedding_vector).tolist()

        # 3. Récupération du centroïde du domaine (pour le boost)
        domain_centroid = query.metadata.get("domain_centroid")

        # 4. Agrégation des scores par journal
        journal_agg: Dict[str, dict] = {}
        
        for art, sim in zip(articles, sims):
            # Récupération du NLM_ID caché dans l'attribut issn (Transport Hack)
            key = getattr(art, "issn", "")
            
            if not key:
                key = getattr(art, "journal_name", "") # Fallback (ex: Scopus)
                
            if key not in journal_agg:
                journal_agg[key] = {
                    "weighted_score": 0.0,
                    "count": 0,
                    "journal_name": art.journal_name,
                    "example_titles": []
                }
                
            # Calcul du Boost Domaine
            boost = 0.0
            if domain_centroid is not None and self._sigaps_repo:
                journal_vec = self._sigaps_repo.get_journal_embedding(key)
                if journal_vec is not None:
                    alignment = max(float(np.dot(domain_centroid, journal_vec)), 0.0)
                    boost = alignment * self._domain_boost_factor

            # Application du score
            journal_agg[key]["weighted_score"] += sim * (1.0 + boost)
            journal_agg[key]["count"] += 1
            
            if len(journal_agg[key]["example_titles"]) < 3:
                journal_agg[key]["example_titles"].append({"title": art.title, "pmid": art.pmid})

        if not journal_agg:
            return []

        # 5. Normalisation et enrichissement SIGAPS
        max_score = max(v["weighted_score"] for v in journal_agg.values()) or 1.0
        
        suggestions: List[SuggestedJournal] = []
        for key, agg in journal_agg.items():
            rank_data = self._sigaps_repo.get_info(key) if self._sigaps_repo else {}
            normalized_sim = round(agg["weighted_score"] / max_score, 4)
            
            suggestions.append(
                SuggestedJournal(
                    nlm_id=rank_data.get("nlm_id", "—"),
                    journal_name=agg["journal_name"],
                    medline_ta=rank_data.get("medline_ta", "—"),
                    similarity_score=normalized_sim,
                    matched_articles_count=agg["count"],
                    rank_sigaps=rank_data.get("rank", "NC"),
                    rank_source=rank_data.get("rank_source", "heuristique"),
                    impact_factor=rank_data.get("impact_factor", ""),
                    medline_indexed=rank_data.get("medline_indexed", ""),
                    example_titles=agg["example_titles"]
                )
            )

        # 6. Tri final
        suggestions.sort(key=lambda x: -x.similarity_score)
        
        # On retourne un large pool de 50 journaux au Frontend
        return suggestions[:max_journals]