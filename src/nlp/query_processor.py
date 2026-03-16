# Fichier : src/nlp/query_processor.py

import re
import numpy as np
from typing import List, Optional

from src.core.entities import ProcessedQuery
from src.core.interfaces import QueryProcessor
from src.nlp.domain_inference import HierarchicalDomainInference

class E5QueryProcessor(QueryProcessor):
    """
    Processeur de requête utilisant un modèle d'embedding dense (ex: E5) 
    et une extraction de tokens basée sur la spécificité.
    """
    
    # On met les sets en majuscules au niveau de la classe (constantes)
    _MEDICAL_STOPWORDS = frozenset({"a", "an", "the", "of", "in", "on", "study", "analysis", "patient", "patients", "effect", "effects", "impact", "role", "use", "using"}) # Ajoute le reste de ta liste ici
    _GENERIC_RESEARCH_TERMS = frozenset({"conditions", "condition", "settings", "setting", "accuracy", "sensitivity", "method", "methods"}) # Ajoute le reste de ta liste ici

    def __init__(self, embedder, domain_engine: Optional[HierarchicalDomainInference] = None):
        """
        Injection de dépendances (Dependency Injection).
        Le processeur n'instancie pas le modèle IA, on lui donne (Inversion de contrôle).
        """
        self._embedder = embedder
        self._domain_engine = domain_engine

    def process(self, title: str) -> ProcessedQuery:
        """Implémente le contrat QueryProcessor."""
        clean_title = re.sub(r"[®©™°†‡§¶]", "", title).strip()
        
        # 1. Extraction lexicale intelligente (sans requêtes booléennes)
        keywords = self._extract_highly_specific_tokens(clean_title)
        
        # 2. Encodage Sémantique (Vecteur)
        embedding_vector = self._embed_text(clean_title)
        
        # 3. Inférence de domaine (Optionnelle)
        metadata = {}
        if self._domain_engine:
            try:
                report = self._domain_engine.predict(clean_title)
                if not report.is_out_of_scope and report.best:
                    metadata["domain_id"] = report.best.domain.id
                    metadata["is_oncology_specialist"] = report.routed_to_specialist
                    # On stocke le centroïde du domaine pour le "boost" ultérieur
                    if report.routed_to_specialist:
                        metadata["domain_centroid"] = self._domain_engine._oncology_centroids.get(report.best.domain.id)
            except Exception as e:
                pass # Fail gracefully, le NLP ne doit pas crasher l'appli

        return ProcessedQuery(
            original_title=clean_title,
            search_keywords=keywords,
            embedding_vector=embedding_vector,
            metadata=metadata
        )

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Génère le vecteur via le modèle injecté."""
        if not self._embedder:
            return None
        # On suppose que ton embedder (ONNX ou SentenceTransformer) expose une méthode encode
        try:
            # Adaptation selon ton wrapper E5 : prefix="query: "
            embs = self._embedder.encode([f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True)
            return embs[0].astype(np.float32)
        except Exception:
            return None

    def _extract_highly_specific_tokens(self, title: str, max_tokens: int = 5) -> List[str]:
        """Extrait les tokens avec une Regex tolérante aux chiffres et symboles (PM2.5, NO2, BRCA1)."""
        # Capture les mots, y compris avec points et tirets (ex: PM2.5, COVID-19)
        tokens = re.findall(r"[a-zA-Z0-9\.\-]+", title)
        kept = []
        
        for tok in tokens:
            tok = tok.strip('.') # Enlève les points finaux
            if not tok: continue
            
            lower = tok.lower()
            if lower in self._MEDICAL_STOPWORDS:
                continue
                
            # RÈGLE 1 : Acronymes, gènes ou formules (PM10, NO2, HER2)
            # Contient au moins une majuscule ET (soit un chiffre, soit une autre majuscule)
            is_acronym_or_chem = bool(re.search(r"[A-Z]", tok) and (re.search(r"[0-9]", tok) or sum(1 for c in tok if c.isupper()) > 1))
            
            # RÈGLE 2 : Mots normaux assez longs (on ignore les simples années comme '1990')
            is_valid_word = len(tok) >= 5 and not tok.isdigit()
            
            if is_acronym_or_chem or is_valid_word:
                kept.append(tok)
                
        # Tri par spécificité
        kept_sorted = sorted(kept, key=self._term_specificity_score, reverse=True)
        return kept_sorted[:max_tokens]

    def _term_specificity_score(self, token: str) -> float:
        """Score les formules/acronymes en priorité absolue."""
        is_acronym = bool(re.search(r"[A-Z]", token) and (re.search(r"[0-9]", token) or sum(1 for c in token if c.isupper()) > 1))
        is_generic = token.lower() in self._GENERIC_RESEARCH_TERMS

        if is_acronym: base = 3.0
        elif not is_generic: base = 2.0
        else: base = 1.0

        return base + len(token) / 1000.0