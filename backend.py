"""
Moteur principal d'OpenSIGAPS.

Ce module gère la logique métier de la plateforme :
- Calcul des scores SIGAPS (Présence et Fractionnaire) selon les règles MERRI 2022.
- Gestion du référentiel des journaux (SigapsRefDB).
- Orchestration de la recherche fédérée (PubMed, Scopus) et déduplication.

"""

import csv
import html
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    ORTModelForFeatureExtraction = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


ONNX_MODEL_DIR = str(Path(__file__).parent / "models" / "bge-m3-onnx-int8")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class _ONNXEmbedder:
    """
    Wrapper ONNX Runtime natif. Chargement : ~400ms. Inférence : ~8ms/phrase.
    """

    def __init__(self, model_dir: str) -> None:
        import onnxruntime as ort

        model_dir_path = Path(model_dir)
        quantized = model_dir_path / "model_quantized.onnx"
        standard = model_dir_path / "model.onnx"
        if quantized.exists():
            model_path = str(quantized)
        elif standard.exists():
            model_path = str(standard)
        else:
            raise FileNotFoundError(f"Aucun fichier .onnx trouvé dans {model_dir}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._out_name = self._session.get_outputs()[0].name
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> "np.ndarray":
        all_parts: list[np.ndarray] = []
        _input_names = [inp.name for inp in self._session.get_inputs()]
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            feeds = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in _input_names:
                feeds["token_type_ids"] = encoded.get(
                    "token_type_ids",
                    np.zeros_like(encoded["input_ids"], dtype=np.int64),
                )
            outputs = self._session.run([self._out_name], feeds)
            token_emb = outputs[0].astype(np.float32)
            attn_mask = encoded["attention_mask"].astype(np.float32)
            mask_exp = attn_mask[:, :, np.newaxis]
            sum_emb = (token_emb * mask_exp).sum(axis=1)
            sum_mask = mask_exp.sum(axis=1).clip(1e-9, None)
            embeddings = (sum_emb / sum_mask).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(
                    1e-9, None
                )
                embeddings = embeddings / norms
            all_parts.append(embeddings)
        return np.vstack(all_parts)


def load_embed_model(onnx_dir: str = ONNX_MODEL_DIR):
    if _ONNX_AVAILABLE and Path(onnx_dir).exists():
        try:
            model = _ONNXEmbedder(onnx_dir)
            logger.info("Modèle chargé via ONNX Runtime depuis %s", onnx_dir)
            return model, "onnx"
        except Exception as e:
            logger.warning("Échec chargement ONNX (%s), tentative PyTorch", e)
    if _ST_AVAILABLE:
        try:
            model = SentenceTransformer("BAAI/bge-m3")
            logger.info("Modèle chargé via sentence-transformers (PyTorch)")
            return model, "pytorch"
        except Exception as e:
            logger.warning("Échec chargement PyTorch : %s", e)
    logger.warning("Aucun modèle d'embedding disponible — fallback Jaccard")
    return None, "jaccard"


VALEUR_POINT_EUROS = 650
REQUEST_TIMEOUT = 30
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")
SCOPUS_API_KEY: str = os.getenv("SCOPUS_API_KEY", "")

C1_COEFFICIENTS: dict[str, int] = {
    "1er": 4,
    "2ème": 3,
    "3ème": 2,
    "ADA": 3,
    "Dernier": 4,
    "Autre": 1,
}

C2_COEFFICIENTS: dict[str, int] = {
    "A+": 14,
    "A": 8,
    "B": 6,
    "C": 4,
    "D": 3,
    "E": 2,
    "NC": 1,
}

SIGAPS_MATRIX: dict[str, dict[str, float]] = {
    pos: {
        rank: float(C1_COEFFICIENTS[pos] * C2_COEFFICIENTS[rank])
        for rank in C2_COEFFICIENTS
    }
    for pos in C1_COEFFICIENTS
}

JOURNALS_A_PLUS: set[str] = {
    "new england journal of medicine",
    "nejm",
    "journal of the american medical association",
    "jama",
    "the british medical journal",
    "british medical journal",
    "bmj",
    "the lancet",
    "lancet",
    "nature",
    "science",
}

RANK_KEYWORDS: dict[str, list[str]] = {
    "A+": list(JOURNALS_A_PLUS),
    "A": [
        "nature medicine",
        "nature genetics",
        "nature communications",
        "blood",
        "cell",
        "immunity",
        "journal of clinical oncology",
        "annals of internal medicine",
        "gut",
        "hepatology",
        "circulation",
        "journal of hepatology",
        "journal of hematology & oncology",
        "american journal of respiratory and critical care",
    ],
    "B": [
        "leukemia",
        "haematologica",
        "annals of oncology",
        "cancer research",
        "cancer",
        "oncology",
        "frontiers in",
        "clinical cancer research",
        "bone marrow transplantation",
        "american journal of hematology",
        "british journal of haematology",
        "european journal of cancer",
    ],
    "D": ["case report", "case reports"],
}


def _trigrams(s: str) -> set[str]:
    s = " " + re.sub(r"[^a-z0-9 ]", " ", s.lower()) + " "
    s = re.sub(r"\s+", " ", s)
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SigapsRefDB:
    """
    Lit sigaps_ref.csv avec csv.DictReader. Pré-calcule les trigrammes pour fuzzy_search O(n).
    Colonnes minimales : NLM_ID, Journal, Latest_Rank.
    """

    VALID_RANKS = {"A+", "A", "B", "C", "D", "E", "NC"}

    def __init__(self) -> None:
        self._rank_by_nlm: dict[str, str] = {}
        self._if_by_nlm: dict[str, str] = {}
        self._medline_by_nlm: dict[str, str] = {}
        self._info_by_nlm: dict[str, dict] = {}
        self._history_by_nlm: dict[str, dict] = {}
        self._if_cols_years: list[tuple] = []
        self.loaded = False
        self.source_path = ""
        self.row_count = 0
        self._emb_matrix: Optional["np.ndarray"] = None
        self._emb_nlm_ids: list[str] = []
        self.embeddings_ready: bool = False

    def load(self, csv_path: "str | Path") -> "SigapsRefDB":
        path = Path(csv_path)
        if not path.exists():
            logger.warning("sigaps_ref.csv introuvable : %s", path)
            return self
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames:
                    reader.fieldnames = [c.strip() for c in reader.fieldnames]
                self._ingest(reader)
        except Exception as exc:
            logger.error("Erreur lecture sigaps_ref.csv : %s", exc)
            return self
        self.loaded = True
        self.source_path = str(path)
        logger.info(
            "SigapsRefDB : %d entrées chargées depuis %s", self.row_count, path.name
        )
        return self

    def _ingest(self, reader: csv.DictReader) -> None:
        fields = reader.fieldnames or []

        def _find(candidates: list[str]) -> Optional[str]:
            low = [f.lower() for f in fields]
            for c in candidates:
                if c.lower() in low:
                    return fields[low.index(c.lower())]
            return None

        rank_col = _find(["latest_rank", "rank"])
        nlm_col = _find(["nlm_id", "nlmid", "nlm id", "nlm_unique_id"])
        journal_col = _find(["journal"])
        medline_col = _find(
            ["indexation medline", "indexation_medline", "medline", "medline_indexed"]
        )

        all_if_cols = sorted(
            [f for f in fields if f.upper().endswith("_IF") or f.upper() == "IF"],
            reverse=True,
        )
        if_col = all_if_cols[0] if all_if_cols else None

        if_year_cols: list[tuple[int, str]] = []
        for col in all_if_cols:
            m = re.match(r"^(\d{4})_IF$", col, re.IGNORECASE)
            if m:
                if_year_cols.append((int(m.group(1)), col))

        rank_year_cols: list[tuple[int, str]] = []
        for f in fields:
            m = re.match(r"^(\d{4})_Rank$|^(\d{4})_Rang$", f, re.IGNORECASE)
            if m:
                rank_year_cols.append((int(m.group(1) or m.group(2)), f))

        if not (rank_col and nlm_col):
            logger.error(
                "CSV : colonnes NLM_ID ou rank introuvables. Colonnes : %s", fields
            )
            return

        # Log des colonnes non reconnues — utile pour détecter les
        # nouvelles colonnes ajoutées au CSV sans casser le chargement.
        _known_patterns = {"nlm", "rank", "journal", "medline", "if", "rang", "issn"}
        _unknown_cols = [
            f
            for f in fields
            if not any(kw in f.lower() for kw in _known_patterns)
            and not re.match(r"^\d{4}_", f)  # colonnes YYYY_IF / YYYY_Rank acceptées
        ]
        if _unknown_cols:
            logger.info(
                "SigapsRefDB : %d colonne(s) non reconnue(s) — ignorées "
                "(compatibilité future assurée) : %s",
                len(_unknown_cols),
                _unknown_cols,
            )

        # Boucle principale — chaque ligne est isolée dans un try/except.
        # Une ligne corrompue (valeur None inattendue, encodage cassé…) est
        # simplement ignorée sans interrompre le chargement du reste du CSV.
        _skipped_rows: int = 0
        for row in reader:
            try:
                nlm = (row.get(nlm_col) or "").strip()
                rank = (row.get(rank_col) or "").strip().upper()
                jrnl = (row.get(journal_col) or "").strip() if journal_col else ""
                if_v = ""
                if if_col:
                    raw = (row.get(if_col) or "").strip()
                    if_v = raw.replace(",", ".") if raw else ""

                medline_v = ""
                if medline_col:
                    medline_v = (row.get(medline_col) or "").strip()

                if not nlm or rank not in self.VALID_RANKS:
                    continue

                tri = _trigrams(jrnl)
                self._rank_by_nlm[nlm] = rank
                self._if_by_nlm[nlm] = if_v
                self._medline_by_nlm[nlm] = medline_v
                self._info_by_nlm[nlm] = {
                    "journal": jrnl,
                    "rank": rank,
                    "if": if_v,
                    "medline": medline_v,
                    "trigrams": tri,
                }

                hist: dict[int, dict] = {}
                for yr, col in if_year_cols:
                    raw = (row.get(col) or "").strip().replace(",", ".")
                    if raw:
                        hist.setdefault(yr, {})["if"] = raw
                for yr, col in rank_year_cols:
                    raw = (row.get(col) or "").strip().upper()
                    if raw in self.VALID_RANKS:
                        hist.setdefault(yr, {})["rank"] = raw
                if hist:
                    self._history_by_nlm[nlm] = hist

                self.row_count += 1

            except Exception as exc:
                # Ligne corrompue — on logue en debug et on continue
                _skipped_rows += 1
                logger.debug("SigapsRefDB : ligne ignorée sur exception (%s)", exc)
                continue

        if _skipped_rows:
            logger.warning(
                "SigapsRefDB : %d ligne(s) ignorée(s) sur erreur de lecture — "
                "vérifiez l'encodage ou les valeurs manquantes du CSV.",
                _skipped_rows,
            )

    def get_rank(self, nlm_id: str) -> Optional[str]:
        return self._rank_by_nlm.get(nlm_id.strip()) if nlm_id else None

    def get_if(self, nlm_id: str) -> str:
        return self._if_by_nlm.get(nlm_id.strip(), "") if nlm_id else ""

    def get_medline(self, nlm_id: str) -> str:
        return self._medline_by_nlm.get(nlm_id.strip(), "") if nlm_id else ""

    def get_info(self, nlm_id: str) -> Optional[dict]:
        return self._info_by_nlm.get(nlm_id.strip()) if nlm_id else None

    def get_history(self, nlm_id: str) -> dict:
        return self._history_by_nlm.get(nlm_id.strip(), {}) if nlm_id else {}

    def build_embeddings(
        self, model, cache_dir: Optional["str | Path"] = None, progress_callback=None
    ) -> bool:
        if not self.loaded or model is None:
            return False
        base = Path(cache_dir or self.source_path).with_suffix("")
        path_emb = Path(str(base) + ".emb.npy")
        path_ids = Path(str(base) + ".emb.ids")
        if self._try_load_cache(path_emb, path_ids):
            logger.info(
                "SigapsRefDB : embeddings chargés depuis cache (%d vecteurs)",
                len(self._emb_nlm_ids),
            )
            self.embeddings_ready = True
            if progress_callback:
                progress_callback(100)
            return True
        logger.info("SigapsRefDB : calcul embeddings pour %d journaux…", self.row_count)
        nlm_ids = list(self._info_by_nlm.keys())
        names = [self._info_by_nlm[n]["journal"] for n in nlm_ids]
        BATCH = 256
        chunks = [names[i : i + BATCH] for i in range(0, len(names), BATCH)]
        parts = []
        for idx, chunk in enumerate(chunks):
            parts.append(_e5_encode(model, chunk, prefix="passage", normalize=True))
            if progress_callback:
                progress_callback(int((idx + 1) / len(chunks) * 95))
        matrix = np.vstack(parts).astype(np.float32)
        try:
            np.save(path_emb, matrix)
            Path(path_ids).write_text("\n".join(nlm_ids), encoding="utf-8")
            logger.info("SigapsRefDB : cache embeddings sauvegardé → %s", path_emb)
        except Exception as e:
            logger.warning("SigapsRefDB : impossible de sauvegarder le cache : %s", e)
        self._emb_matrix = matrix
        self._emb_nlm_ids = nlm_ids
        self.embeddings_ready = True
        if progress_callback:
            progress_callback(100)
        return True

    def _try_load_cache(self, path_emb: Path, path_ids: Path) -> bool:
        if not path_emb.exists() or not path_ids.exists():
            return False
        try:
            matrix = np.load(path_emb)
            nlm_ids = Path(path_ids).read_text(encoding="utf-8").splitlines()
            if len(nlm_ids) != self.row_count:
                logger.info(
                    "Cache invalidé (taille %d ≠ CSV %d)", len(nlm_ids), self.row_count
                )
                return False
            self._emb_matrix = matrix.astype(np.float32)
            self._emb_nlm_ids = nlm_ids
            return True
        except Exception as e:
            logger.warning("Cache embeddings illisible : %s", e)
            return False

    def semantic_search(self, query_vec: "np.ndarray", top_k: int = 8) -> list[dict]:
        if not self.embeddings_ready or self._emb_matrix is None:
            return []
        scores = self._emb_matrix @ query_vec
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = []
        for i in top_idx:
            nlm_id = self._emb_nlm_ids[i]
            info = self._info_by_nlm.get(nlm_id, {})
            results.append(
                {
                    "nlm_id": nlm_id,
                    "journal": info.get("journal", ""),
                    "rank": info.get("rank", "NC"),
                    "if": info.get("if", ""),
                    "medline": info.get("medline", ""),
                    "similarity": float(scores[i]),
                }
            )
        return results

    def fuzzy_search(
        self, query: str, top_k: int = 8, threshold: float = 0.08
    ) -> list[dict]:
        if not self.loaded:
            return []
        q_tri = _trigrams(query)
        if not q_tri:
            return []
        scored: list[tuple[float, str, dict]] = [
            (_jaccard(q_tri, info["trigrams"]), nlm_id, info)
            for nlm_id, info in self._info_by_nlm.items()
            if _jaccard(q_tri, info["trigrams"]) >= threshold
        ]
        scored.sort(key=lambda x: -x[0])
        return [
            {
                "nlm_id": nlm_id,
                "journal": info["journal"],
                "rank": info["rank"],
                "if": info["if"],
                "medline": info.get("medline", ""),
                "similarity": sim,
            }
            for sim, nlm_id, info in scored[:top_k]
        ]

    def __len__(self) -> int:
        return self.row_count


_REF_DB: SigapsRefDB = SigapsRefDB()
_EMBED_MODEL = None
_EMBED_BACKEND = "jaccard"


def set_embed_model(model, backend: str = "jaccard") -> None:
    global _EMBED_MODEL, _EMBED_BACKEND
    _EMBED_MODEL = model
    _EMBED_BACKEND = backend


def get_embed_model():
    return _EMBED_MODEL


def get_embed_backend() -> str:
    return _EMBED_BACKEND


def _e5_encode(
    model,
    texts: list[str],
    prefix: str = "query",  # Conservé dans la signature pour la rétrocompatibilité
    batch_size: int = 256,
    normalize: bool = True,
) -> "np.ndarray":
    """
    Encode les textes avec BAAI/bge-m3. 
    Note d'architecture : Ce modèle ne nécessite PAS de préfixes (contrairement à e5 utilisé précedemment) 
    L'argument 'prefix' est donc volontairement ignoré ici.
    """
    embs = model.encode(
        texts,  # On passe directement les textes bruts, sans le f"{prefix}: "
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def set_ref_db(db: SigapsRefDB) -> None:
    global _REF_DB
    _REF_DB = db


def get_ref_db() -> SigapsRefDB:
    return _REF_DB


def calculate_presence_score(position: str, rank: str) -> float:
    return SIGAPS_MATRIX.get(position, {}).get(rank, 0.0)


def calculate_sum_c1(nb_authors: int) -> float:
    """
    Calcule Σ C1 (dénominateur score fractionnaire MERRI 2022).
      n=1→4, n=2→8, n=3→11, n=4→13, n=5→14, n≥6→n+11
    """
    if nb_authors <= 0:
        return 1.0
    if nb_authors == 1:
        return 4.0
    if nb_authors == 2:
        return 8.0
    if nb_authors == 3:
        return 11.0
    if nb_authors == 4:
        return 13.0
    if nb_authors == 5:
        return 14.0
    return float(nb_authors + 11)


def calculate_fractional_score(position: str, rank: str, nb_authors: int) -> float:
    c1 = float(C1_COEFFICIENTS.get(position, 1))
    c2 = float(C2_COEFFICIENTS.get(rank, 1))
    sum_c1 = calculate_sum_c1(nb_authors)
    return round((c1 / sum_c1) * c2, 4) if sum_c1 else 0.0


def calculate_team_presence_score(
    articles_by_member: dict[str, list["Article"]],
) -> dict:
    article_pool: dict[str, list[tuple[str, "Article"]]] = {}
    for member_name, arts in articles_by_member.items():
        for art in arts:
            if not art.is_selected:
                continue
            key = art.doi if art.doi else _normalize_for_dedup(art.title)
            if not key or len(key) < 5:
                key = _normalize_for_dedup(art.title)
            article_pool.setdefault(key, []).append((member_name, art))

    results: list[dict] = []
    total_presence = total_frac = 0.0

    for members_arts in article_pool.values():
        sorted_arts = sorted(
            members_arts,
            key=lambda x: C1_COEFFICIENTS.get(x[1].my_position, 1),
            reverse=True,
        )
        best_member, best_art = sorted_arts[0]
        pts_presence = float(
            C1_COEFFICIENTS.get(best_art.my_position, 1)
            * C2_COEFFICIENTS.get(best_art.estimated_rank, 1)
        )
        pts_frac = sum(
            calculate_fractional_score(a.my_position, a.estimated_rank, a.nb_authors)
            for _, a in members_arts
            if a.nb_authors > 0
        )
        results.append(
            {
                "title": best_art.title,
                "journal": best_art.journal_name,
                "year": best_art.publication_year,
                "rank": best_art.estimated_rank,
                "rank_source": best_art.rank_source,
                "best_member": best_member,
                "best_position": best_art.my_position,
                "co_signataires_equipe": [n for n, _ in members_arts],
                "nb_membres_coauteurs": len(members_arts),
                "pts_presence": pts_presence,
                "pts_frac": round(pts_frac, 4),
                "doi": best_art.doi,
            }
        )
        total_presence += pts_presence
        total_frac += pts_frac

    results.sort(key=lambda x: x["year"], reverse=True)
    return {
        "articles_uniques": results,
        "total_pts_presence": round(total_presence, 2),
        "total_pts_frac": round(total_frac, 4),
    }


@dataclass
class Article:
    id: str
    source: str
    title: str
    publication_year: int
    journal_name: str
    authors_list: list[str]
    doi: str = ""
    my_position: str = "Autre"
    estimated_rank: str = "C"
    is_selected: bool = True
    nb_authors: int = 0
    pmid: str = ""
    openalex_id: str = ""
    nlm_unique_id: str = ""
    rank_source: str = "heuristique"
    concepts: str = "" # Stocke les MeSH ou Keywords

    def calculate_points(self) -> float:
        return self.presence_score()

    def presence_score(self) -> float:
        return (
            calculate_presence_score(self.my_position, self.estimated_rank)
            if self.is_selected
            else 0.0
        )

    def fractional_score(self) -> float:
        if not self.is_selected or self.nb_authors == 0:
            return 0.0
        return calculate_fractional_score(
            self.my_position, self.estimated_rank, self.nb_authors
        )


def _normalize_for_dedup(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\W+", "", text.lower())


def _filter_by_affiliation(
    articles: list[Article],
    affiliation: str,
    city: str,
    researcher_name: str,
) -> list[Article]:
    """
    Filtre post-récupération par affiliation et/ou ville.

    Stratégie : appel efetch pour chaque article afin de lire <AffiliationInfo>.
    Pour préserver les performances, on filtre uniquement si le champ est renseigné.
    Les articles sans information d'affiliation dans PubMed sont conservés par défaut
    (politique permissive) pour ne pas silencieusement exclure des vrais positifs.

    Note : les articles source=Scopus ne bénéficient pas encore de ce filtre
    (l'API Scopus retourne les affiliations différemment — extension future).
    """
    if not affiliation and not city:
        return articles

    aff_norm = _normalize_name(affiliation)
    city_norm = _normalize_name(city)
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Récupère uniquement les PMIDs PubMed (Scopus non filtrés)
    pubmed_articles = [a for a in articles if a.pmid]
    scopus_articles = [a for a in articles if not a.pmid]

    if not pubmed_articles:
        return articles

    pmids = [a.pmid for a in pubmed_articles]

    try:
        resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("_filter_by_affiliation efetch : %s — filtre ignoré", e)
        return articles  # politique permissive en cas d'erreur

    # Indexer les affiliations par PMID
    affil_by_pmid: dict[str, list[str]] = {}
    for art_tag in root.findall(".//PubmedArticle"):
        pmid_tag = art_tag.find(".//PMID")
        if pmid_tag is None:
            continue
        pmid = (pmid_tag.text or "").strip()
        affils: list[str] = []
        for affil_tag in art_tag.findall(".//AffiliationInfo/Affiliation"):
            if affil_tag.text:
                affils.append(_normalize_name(affil_tag.text))
        affil_by_pmid[pmid] = affils

    filtered: list[Article] = []
    for art in pubmed_articles:
        affils = affil_by_pmid.get(art.pmid, [])
        if not affils:
            # Pas d'info d'affiliation → on garde (politique permissive)
            filtered.append(art)
            continue
        combined = " ".join(affils)
        match_aff = (not aff_norm) or (aff_norm in combined)
        match_city = (not city_norm) or (city_norm in combined)
        if match_aff and match_city:
            filtered.append(art)

    # Scopus toujours inclus (filtre non implémenté)
    return filtered + scopus_articles


def _normalize_name(name: str) -> str:
    table = str.maketrans("àáâäãéèêëîïíìôöóòùûüúñç", "aaaaaeeeeiiiioooouuuunc")
    return name.lower().translate(table)


def _estimate_rank(journal_name: str) -> str:
    j_norm = _normalize_name(journal_name)
    for rank, keywords in RANK_KEYWORDS.items():
        if any(_normalize_name(kw) in j_norm for kw in keywords):
            return rank
    return "C"


def _resolve_rank(
    journal_name: str, nlm_id: str = "", ref_db: Optional[SigapsRefDB] = None
) -> tuple[str, str]:
    db = ref_db or _REF_DB
    if nlm_id and db.loaded:
        rank = db.get_rank(nlm_id)
        if rank:
            return rank, "csv"
    return _estimate_rank(journal_name), "heuristique"


def _determine_position(idx: int, total_authors: int) -> str:
    n = total_authors
    if idx == 0:
        return "1er"
    if idx == n - 1 and n > 1:
        return "Dernier"  # ← remonté avant "3ème"
    if idx == n - 2 and n >= 6:
        return "ADA"  # ← remonté avant "3ème"
    if idx == 1:
        return "2ème"
    if idx == 2:
        return "3ème"
    return "Autre"


class PubMedFetcher:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, ref_db: Optional[SigapsRefDB] = None) -> None:
        self.ref_db = ref_db or _REF_DB

    def fetch_articles(self, researcher_name: str, limit: int = 500) -> list[Article]:
        try:
            uids, total = self._search_uids_paginated(researcher_name, limit)
            if not uids:
                return []
            if total > limit:
                logger.info(
                    "PubMed : %d articles trouvés pour '%s', limité à %d",
                    total,
                    researcher_name,
                    limit,
                )
            return self._fetch_details_batched(uids, researcher_name)
        except requests.exceptions.SSLError as e:
            logger.warning("PubMed SSL error : %s", e)
            return []
        except requests.RequestException as e:
            logger.warning("PubMed réseau: %s", e)
            return []
        except ET.ParseError as e:
            logger.warning("PubMed XML malformé: %s", e)
            return []

    def _search_uids_paginated(
        self, name: str, limit: int, batch_size: int = 200
    ) -> tuple[list[str], int]:
        all_uids: list[str] = []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": f"{name}[Author]",
                    "retmax": min(batch_size, limit),
                    "retstart": 0,
                    "retmode": "json",
                    **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.SSLError as e:
            logger.warning("PubMed SSL error dans _search_uids_paginated : %s", e)
            return [], 0
        except requests.RequestException as e:
            logger.warning("PubMed réseau dans _search_uids_paginated : %s", e)
            return [], 0

        data = resp.json().get("esearchresult", {})
        total = int(data.get("count", 0))
        all_uids.extend(data.get("idlist", []))

        retstart = len(all_uids)
        while retstart < min(total, limit):
            batch = min(batch_size, limit - retstart)
            r = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": f"{name}[Author]",
                    "retmax": batch,
                    "retstart": retstart,
                    "retmode": "json",
                    **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
                },
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                break
            all_uids.extend(ids)
            retstart += len(ids)

        return all_uids[:limit], total

    def _fetch_details_batched(
        self, uids: list[str], researcher_name: str, batch_size: int = 200
    ) -> list[Article]:
        target_norm = _normalize_name(researcher_name.split()[-1])
        articles: list[Article] = []
        for i in range(0, len(uids), batch_size):
            batch = uids[i : i + batch_size]
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "retmode": "xml",
                        **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                for art_tag in root.findall(".//PubmedArticle"):
                    art = self._parse_article(art_tag, target_norm)
                    if art is not None:
                        articles.append(art)
            except (requests.RequestException, ET.ParseError) as e:
                logger.warning("Batch efetch %d-%d échoué : %s", i, i + batch_size, e)
                continue
        return articles

    def _search_uids(self, name: str, limit: int) -> list[str]:
        uids, _ = self._search_uids_paginated(name, limit)
        return uids

    def _fetch_details(self, uids: list[str], researcher_name: str) -> list[Article]:
        return self._fetch_details_batched(uids, researcher_name)

    def _parse_article(
        self, art_tag: ET.Element, target_last_name: str
    ) -> Optional[Article]:
        title_tag = art_tag.find(".//ArticleTitle")
        pmid_tag = art_tag.find(".//PMID")
        if title_tag is None or pmid_tag is None:
            return None
        title = html.unescape(
            ET.tostring(title_tag, method="text", encoding="unicode")
        ).strip()
        if not title:
            return None
        year_tag = art_tag.find(".//PubDate/Year")
        year = int(year_tag.text) if year_tag is not None else datetime.now().year
        journal_tag = art_tag.find(".//Journal/Title")
        journal = journal_tag.text if journal_tag is not None else "Revue Inconnue"
        nlm_tag = art_tag.find(".//MedlineJournalInfo/NlmUniqueID")
        nlm_id = (nlm_tag.text or "").strip() if nlm_tag is not None else ""
        doi = ""
        for id_tag in art_tag.findall(".//ArticleId"):
            if id_tag.get("IdType") == "doi":
                doi = id_tag.text or ""
                break
        # --- Extraction MeSH & Keywords (Système Fallback) ---
        concepts_list = []
        # Priorité 1 : Les termes MeSH (Pureté)
        mesh_list = art_tag.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if mesh_list:
            concepts_list = [m.text for m in mesh_list if m.text]
        else:
            # Priorité 2 : Author Keywords si pas encore indexé MeSH (Immédiateté)
            kw_list = art_tag.findall(".//KeywordList/Keyword")
            if kw_list:
                concepts_list = [k.text for k in kw_list if k.text]
        
        concepts = " ".join(concepts_list)
        #---
        authors = self._extract_authors(art_tag)
        nb_authors = len(authors)
        position = self._find_position(authors, target_last_name, nb_authors)
        rank, rank_source = _resolve_rank(journal, nlm_id, self.ref_db)
        return Article(
            id=f"PMID:{pmid_tag.text}",
            pmid=pmid_tag.text or "",
            source="PubMed",
            title=title,
            publication_year=year,
            journal_name=journal,
            authors_list=authors,
            doi=doi,
            my_position=position,
            estimated_rank=rank,
            nb_authors=nb_authors,
            nlm_unique_id=nlm_id,
            rank_source=rank_source,
            concepts=concepts,
        )

    @staticmethod
    def _extract_authors(art_tag: ET.Element) -> list[str]:
        authors = []
        for auth in art_tag.findall(".//AuthorList/Author"):
            last = auth.find("LastName")
            initials = auth.find("Initials")
            if last is not None:
                name = last.text or ""
                if initials is not None:
                    name += f" {initials.text}"
                authors.append(name)
        return authors

    @staticmethod
    def _find_position(
        authors: list[str], target_last_name: str, nb_authors: int
    ) -> str:
        for i, name in enumerate(authors):
            if target_last_name in _normalize_name(name):
                return _determine_position(i, nb_authors)
        return "Autre"

    @staticmethod
    def _deduplicate(articles: list[Article]) -> list[Article]:
        seen_dois: set[str] = set()
        seen_pmids: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Article] = []
        for art in articles:
            if art.doi and art.doi in seen_dois:
                continue
            if art.pmid and art.pmid in seen_pmids:
                continue
            norm = _normalize_for_dedup(art.title)
            if len(norm) > 15 and norm in seen_titles:
                continue
            if art.doi:
                seen_dois.add(art.doi)
            if art.pmid:
                seen_pmids.add(art.pmid)
            seen_titles.add(norm)
            unique.append(art)
        return unique

    def search(
        self,
        name: str,
        institution_filter: Optional[str] = None,
        limit_per_source: int = 500,
    ) -> list[Article]:
        articles = self.fetch_articles(name, limit=limit_per_source)
        articles = self._deduplicate(articles)
        articles.sort(key=lambda x: x.publication_year, reverse=True)
        return articles


class ScopusFetcher:
    """
    Interroge l'API Scopus (Elsevier) pour récupérer les articles d'un auteur.
    Endpoint : https://api.elsevier.com/content/search/scopus
    Authentification : header X-ELS-APIKey

    Pagination : paramètres start/count (max 200 par batch, 5000 au total).
    Déduplication cross-source : via DOI (primaire) ou titre normalisé (fallback).
    """

    BASE_URL = "https://api.elsevier.com/content/search/scopus"
    BATCH_SIZE: int = 200

    def __init__(self, ref_db: Optional[SigapsRefDB] = None) -> None:
        self.ref_db = ref_db or _REF_DB
        self._available: bool = bool(SCOPUS_API_KEY)
        if not self._available:
            logger.info("ScopusFetcher : SCOPUS_API_KEY absent — désactivé")

    def _headers(self) -> dict[str, str]:
        return {
            "X-ELS-APIKey": SCOPUS_API_KEY,
            "Accept": "application/json",
        }

    def fetch_articles(
        self,
        researcher_name: str,
        limit: int = 500,
    ) -> list[Article]:
        if not self._available:
            return []
        try:
            return self._paginate(researcher_name, limit)
        except requests.exceptions.RequestException as e:
            logger.warning("ScopusFetcher réseau : %s", e)
            return []
        except Exception as e:
            logger.warning("ScopusFetcher inattendu : %s", e)
            return []

    def _paginate(self, name: str, limit: int) -> list[Article]:
        """Pagination Scopus : start incrémenté par BATCH_SIZE jusqu'à limit."""
        # Formatage du nom pour la requête Scopus : "Fervers B" → AU-NAME(Fervers)
        # L'API Scopus accepte le nom complet : AU-NAME("Béatrice Fervers")
        parts = name.strip().split()
        if len(parts) >= 2:
            # "Romain Buono" → AUTH("Fervers Beatrice") — Scopus préfère Nom Prénom
            query = f'AUTH("{parts[-1]} {" ".join(parts[:-1])}")'
        else:
            # Nom seul → AUTHLASTNAME(Fervers)
            query = f"AUTHLASTNAME({parts[0]})"
        all_articles: list[Article] = []
        start = 0

        while start < limit:
            count = min(self.BATCH_SIZE, limit - start)
            try:
                resp = requests.get(
                    self.BASE_URL,
                    headers=self._headers(),
                    params={
                        "query": query,
                        "count": count,
                        "start": start,
                        "httpAccept": "application/json",
                        "field": (
                            "dc:title,prism:publicationName,prism:doi,"
                            "prism:coverDate,dc:creator,author,prism:issn,"
                            "eid,citedby-count,authkeywords"
                        ),
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning("ScopusFetcher : 429 rate-limit, arrêt pagination")
                    break
                raise

            data = resp.json().get("search-results", {})
            total = int(data.get("opensearch:totalResults", 0))
            entries = data.get("entry", [])

            if not entries:
                break

            for entry in entries:
                art = self._parse_entry(entry, name)
                if art is not None:
                    all_articles.append(art)

            start += len(entries)
            if start >= total:
                break

        logger.info(
            "ScopusFetcher : %d articles récupérés pour '%s'",
            len(all_articles),
            name,
        )
        return all_articles

    def search_by_topic(
        self,
        query_terms: list[str],
        limit: int = 50,
    ) -> list[dict]:
        """
        [v8.2] Recherche thématique Scopus pour la suggestion de journaux.

        Utilise l'endpoint TITLE-ABS-KEY — distinct de AUTH() utilisé dans fetch_articles().
        Ne touche pas à la recherche auteur/équipe existante (SRP).

        Stratégie :
          - Prend jusqu'à 4 termes fournis par domain_inference (les plus spécifiques).
          - Retourne des dicts {art_title, journal_name, issn, doi} utilisés pour
            enrichir le journal_agg de suggest_journals_by_title().
          - En cas d'absence de SCOPUS_API_KEY ou d'erreur réseau : retourne [] sans exception.

        Args:
            query_terms : liste de tokens thématiques (ex: ["lung cancer", "immunotherapy"])
            limit       : nombre max d'articles à récupérer (max 200 par batch Scopus)

        Returns:
            Liste de dicts {art_title, journal_name, issn, doi}.
        """
        if not self._available or not query_terms:
            return []

        # Construction de la requête TITLE-ABS-KEY avec les termes les plus spécifiques
        terms_quoted: list[str] = [f'"{t}"' for t in query_terms[:4]]
        query: str = "TITLE-ABS-KEY(" + " AND ".join(terms_quoted) + ")"

        try:
            resp = requests.get(
                self.BASE_URL,
                headers=self._headers(),
                params={
                    "query": query,
                    "count": min(limit, self.BATCH_SIZE),
                    "start": 0,
                    "httpAccept": "application/json",
                    "field": "dc:title,prism:publicationName,prism:issn,prism:doi",
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            logger.warning("ScopusFetcher.search_by_topic HTTP %s: %s", status, e)
            return []
        except requests.exceptions.RequestException as e:
            logger.warning("ScopusFetcher.search_by_topic réseau: %s", e)
            return []

        entries: list[dict] = resp.json().get("search-results", {}).get("entry", [])
        results: list[dict] = []
        for entry in entries:
            journal_name: str = (entry.get("prism:publicationName") or "").strip()
            if not journal_name:
                continue
            results.append(
                {
                    "art_title": (entry.get("dc:title") or "").strip(),
                    "journal_name": journal_name,
                    "issn": (entry.get("prism:issn") or "").strip(),
                    "doi": (entry.get("prism:doi") or "").strip(),
                }
            )

        logger.info(
            "ScopusFetcher.search_by_topic : %d articles pour %r",
            len(results),
            query[:80],
        )
        return results

    def _parse_entry(self, entry: dict, researcher_name: str) -> Optional[Article]:
        title = (entry.get("dc:title") or "").strip()
        if not title:
            return None

        # Date de publication
        cover_date: str = entry.get("prism:coverDate") or ""
        try:
            year = int(cover_date[:4]) if cover_date else datetime.now().year
        except ValueError:
            year = datetime.now().year

        journal = (entry.get("prism:publicationName") or "Revue Inconnue").strip()
        doi = (entry.get("prism:doi") or "").strip()
        eid = (entry.get("eid") or "").strip()
        # --- Extraction Author Keywords Scopus ---
            # Scopus renvoie souvent une chaîne séparée par des "|"
        raw_keywords = entry.get("authkeywords", "")
        concepts = raw_keywords.replace("|", " ").strip() if raw_keywords else ""
        # ---

        # Auteurs
        raw_authors = entry.get("author", [])
        authors: list[str] = []
        for a in raw_authors:
            surname = (a.get("surname") or "").strip()
            given_init = (a.get("initials") or "").strip()
            if surname:
                authors.append(f"{surname} {given_init}".strip())

        nb_authors = len(authors)
        target_norm = _normalize_name(researcher_name.split()[-1])
        position = PubMedFetcher._find_position(authors, target_norm, nb_authors)
        rank, rank_source = _resolve_rank(journal, "", self.ref_db)

        return Article(
            id=f"SCOPUS:{eid}",
            source="Scopus",
            title=title,
            publication_year=year,
            journal_name=journal,
            authors_list=authors,
            doi=doi,
            my_position=position,
            estimated_rank=rank,
            rank_source=rank_source,
            nb_authors=nb_authors,
            concepts=concepts,
        )


class FederatedSearch:
    """
    Orchestrateur multi-source : PubMed + Scopus.
    Déduplication par DOI (primaire) puis titre normalisé (fallback).
    Expose la même interface que PubMedFetcher pour compatibilité app.py.
    """

    def __init__(self, ref_db: Optional[SigapsRefDB] = None) -> None:
        self._pubmed = PubMedFetcher(ref_db=ref_db)
        self._scopus = ScopusFetcher(ref_db=ref_db)
        self.ref_db = ref_db or _REF_DB

    # ── Délégation des méthodes PubMed-only utilisées par app.py ─────────────
    def _search_uids_paginated(self, name: str, limit: int) -> tuple[list[str], int]:
        return self._pubmed._search_uids_paginated(name, limit)

    def _fetch_details_batched(self, uids: list[str], name: str) -> list[Article]:
        return self._pubmed._fetch_details_batched(uids, name)

    def _deduplicate(self, articles: list[Article]) -> list[Article]:
        return PubMedFetcher._deduplicate(articles)

    # ── Recherche fédérée (Individuel + Équipe) ───────────────────────────────
    def search(
        self,
        name: str,
        institution_filter: Optional[str] = None,
        limit_per_source: int = 500,
    ) -> list[Article]:
        """
        Fusionne PubMed et Scopus. Déduplication :
          1. DOI identique → un seul article conservé (PubMed prioritaire)
          2. Titre normalisé identique → PubMed conservé
        """
        pubmed_arts = self._pubmed.fetch_articles(name, limit=limit_per_source)
        scopus_arts = self._scopus.fetch_articles(name, limit=limit_per_source)

        merged = self._merge(pubmed_arts, scopus_arts)
        merged.sort(key=lambda x: x.publication_year, reverse=True)
        return merged

    @staticmethod
    def _merge(primary: list[Article], secondary: list[Article]) -> list[Article]:
        """
        Fusionne deux listes d'articles. PubMed (primary) est prioritaire.
        Une entrée Scopus est ignorée si son DOI ou titre normalisé existe déjà.
        """
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        result: list[Article] = []

        for art in primary:
            if art.doi:
                seen_dois.add(art.doi.lower())
            norm = _normalize_for_dedup(art.title)
            if len(norm) > 15:
                seen_titles.add(norm)
            result.append(art)

        for art in secondary:
            doi_key = art.doi.lower() if art.doi else ""
            title_key = _normalize_for_dedup(art.title)

            if doi_key and doi_key in seen_dois:
                continue  # doublon DOI
            if len(title_key) > 15 and title_key in seen_titles:
                continue  # doublon titre

            if doi_key:
                seen_dois.add(doi_key)
            if len(title_key) > 15:
                seen_titles.add(title_key)
            result.append(art)

        return result


@dataclass
class JournalResult:
    nlm_id: str
    journal_name: str
    medline_ta: str
    rank: str
    rank_source: str
    impact_factor: str = ""
    medline_indexed: str = ""
    issn: str = ""
    country: str = ""
    similarity: float = 0.0


def fetch_article_by_pmid(
    pmid: str, ref_db: Optional[SigapsRefDB] = None
) -> Optional[dict]:
    db = ref_db or _REF_DB
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": pmid.strip(),
                "retmode": "xml",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        art_tag = root.find(".//PubmedArticle")
        if art_tag is None:
            return None
        title_tag = art_tag.find(".//ArticleTitle")
        title = (
            html.unescape(
                ET.tostring(title_tag, method="text", encoding="unicode")
            ).strip()
            if title_tag is not None
            else "—"
        )
        journal_tag = art_tag.find(".//Journal/Title")
        journal = journal_tag.text if journal_tag is not None else "—"
        mta_tag = art_tag.find(".//MedlineJournalInfo/MedlineTA")
        medline_ta = mta_tag.text if mta_tag is not None else "—"
        nlm_tag = art_tag.find(".//MedlineJournalInfo/NlmUniqueID")
        nlm_id = (nlm_tag.text or "").strip() if nlm_tag is not None else ""
        year_tag = art_tag.find(".//PubDate/Year")
        year = int(year_tag.text) if year_tag is not None else None
        doi = ""
        for id_tag in art_tag.findall(".//ArticleId"):
            if id_tag.get("IdType") == "doi":
                doi = id_tag.text or ""
                break
        authors = []
        for auth in art_tag.findall(".//AuthorList/Author"):
            last = auth.find("LastName")
            initials = auth.find("Initials")
            if last is not None:
                a = (last.text or "") + (
                    " " + (initials.text or "") if initials is not None else ""
                )
                authors.append(a.strip())
        rank, rank_source = _resolve_rank(journal, nlm_id, db)
        impact_factor = db.get_if(nlm_id) if db.loaded else ""
        return {
            "title": title,
            "journal": journal,
            "medline_ta": medline_ta,
            "nlm_id": nlm_id,
            "rank": rank,
            "rank_source": rank_source,
            "impact_factor": impact_factor,
            "year": year,
            "doi": doi,
            "authors": authors,
        }
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("fetch_article_by_pmid(%s): %s", pmid, e)
        return None


def fetch_article_by_doi(
    doi: str, ref_db: Optional[SigapsRefDB] = None
) -> Optional[dict]:
    db = ref_db or _REF_DB
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    doi = (
        doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/").lstrip("doi:")
    )
    try:
        s_resp = requests.get(
            f"{BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{doi}[AID]",
                "retmax": 1,
                "retmode": "json",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        s_resp.raise_for_status()
        ids = s_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            logger.warning("fetch_article_by_doi: aucun PMID pour DOI %s", doi)
            return None
        return fetch_article_by_pmid(ids[0], ref_db=db)
    except (requests.RequestException, Exception) as e:
        logger.warning("fetch_article_by_doi(%s): %s", doi, e)
        return None


def _nlm_catalog_search(query: str, max_results: int = 8) -> list[dict]:
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        s_resp = requests.get(
            f"{BASE}/esearch.fcgi",
            params={
                "db": "nlmcatalog",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        s_resp.raise_for_status()
        uids = s_resp.json().get("esearchresult", {}).get("idlist", [])
        if not uids:
            return []
        f_resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={
                "db": "nlmcatalog",
                "id": ",".join(uids),
                "retmode": "xml",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        f_resp.raise_for_status()
        root = ET.fromstring(f_resp.content)
        results = []
        for rec in root.findall(".//NLMCatalogRecord"):
            nlm_tag = rec.find(".//NlmUniqueID")
            nlm_id = (nlm_tag.text or "").strip() if nlm_tag is not None else ""
            title_tag = rec.find(".//TitleMain/Title")
            full_title = title_tag.text.strip() if title_tag is not None else "—"
            mta_tag = rec.find(".//MedlineTA")
            medline_ta = mta_tag.text if mta_tag is not None else "—"
            issn_tag = rec.find(".//ISSN")
            issn = issn_tag.text if issn_tag is not None else ""
            cty_tag = rec.find(".//Country")
            country = cty_tag.text if cty_tag is not None else ""
            results.append(
                {
                    "nlm_id": nlm_id,
                    "journal": full_title,
                    "medline_ta": medline_ta,
                    "issn": issn,
                    "country": country,
                }
            )
        return results
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("_nlm_catalog_search('%s'): %s", query, e)
        return []


def search_journal_by_name(
    query: str, ref_db: Optional[SigapsRefDB] = None, max_results: int = 8
) -> list[JournalResult]:
    db = ref_db or _REF_DB
    model = _EMBED_MODEL

    query_vec: Optional[np.ndarray] = None
    if model is not None and db.embeddings_ready:
        query_vec = _e5_encode(model, [query], prefix="query")[0]

    def _csv_search():
        if query_vec is not None:
            return db.semantic_search(query_vec, top_k=max_results)
        elif db.loaded:
            return db.fuzzy_search(query, top_k=max_results)
        return []

    with ThreadPoolExecutor(max_workers=2) as exe:
        f_api = exe.submit(_nlm_catalog_search, query, max_results)
        f_csv = exe.submit(_csv_search)
        api_raw = f_api.result()
        csv_raw = f_csv.result()

    merged: dict[str, JournalResult] = {}
    for r in api_raw:
        nlm_id = r["nlm_id"]
        rank, rank_source = _resolve_rank(r["journal"], nlm_id, db)
        impact_factor = db.get_if(nlm_id) if db.loaded else ""
        medline_indexed = db.get_medline(nlm_id) if db.loaded else ""
        if query_vec is not None:
            r_vec = _e5_encode(model, [r["journal"]], prefix="passage")[0]
            sim = float(np.dot(query_vec, r_vec))
        else:
            sim = _jaccard(_trigrams(query), _trigrams(r["journal"]))
        merged[nlm_id] = JournalResult(
            nlm_id=nlm_id,
            journal_name=r["journal"],
            medline_ta=r["medline_ta"],
            rank=rank,
            rank_source=rank_source,
            impact_factor=impact_factor,
            medline_indexed=medline_indexed,
            issn=r["issn"],
            country=r["country"],
            similarity=sim,
        )

    for r in csv_raw:
        nlm_id = r["nlm_id"]
        if nlm_id in merged:
            if r["similarity"] > merged[nlm_id].similarity:
                merged[nlm_id].similarity = r["similarity"]
            continue
        merged[nlm_id] = JournalResult(
            nlm_id=nlm_id,
            journal_name=r["journal"],
            medline_ta="—",
            rank=r["rank"],
            rank_source="csv",
            impact_factor=r["if"],
            medline_indexed=r.get("medline", ""),
            issn="",
            country="",
            similarity=r["similarity"],
        )

    return sorted(merged.values(), key=lambda x: -x.similarity)[:max_results]


def detect_article_domain(
    article_title: str,
    domain_engine,  # Optional[HierarchicalDomainInference]
) -> Optional[dict]:
    """
    [v8.2] Détecte le domaine sémantique d'un titre via HierarchicalDomainInference.

    Fonction utilitaire publique appelée par app.py pour afficher le badge de domaine
    SANS dupliquer l'appel d'inférence effectué dans suggest_journals_by_title().

    Returns:
        dict {domain_id, family, confidence, raw_score, margin,
              routed_to_specialist, oncology_meta_score}
        ou None si hors domaine / moteur indisponible / erreur.
    """
    if domain_engine is None:
        return None
    try:
        report = domain_engine.predict(article_title)
        if report.is_out_of_scope or not report.best:
            return None
        return {
            "domain_id": report.best.domain.id,
            "family": report.best.domain.family.value,
            "confidence": round(report.best.confidence, 3),
            "raw_score": round(report.best.raw_score, 3),
            "margin": round(report.best.margin, 3),
            "routed_to_specialist": report.routed_to_specialist,
            "oncology_meta_score": round(report.oncology_meta_score, 3),
        }
    except Exception as exc:
        logger.warning("detect_article_domain: %s — ignoré", exc)
        return None


@dataclass
class SuggestedJournal:
    """Journal suggéré par analyse des articles similaires PubMed."""

    nlm_id: str
    journal_name: str
    medline_ta: str
    rank: str
    rank_source: str
    impact_factor: str
    medline_indexed: str
    issn: str
    country: str
    similarity: float
    article_count: int
    example_titles: list
    query_level: int = 1
    query_used: str = ""


# ── Stopwords d'exclusion stricte ─────────────────────────────────────────────
# Termes filtrés AVANT d'entrer dans le pipeline de requête.

_MEDICAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "with",
        "without",
        "by",
        "from",
        "into",
        "through",
        "after",
        "before",
        "between",
        "among",
        "versus",
        "vs",
        "per",
        "effect",
        "effects",
        "impact",
        "role",
        "roles",
        "use",
        "using",
        "study",
        "studies",
        "analysis",
        "evaluation",
        "assessment",
        "review",
        "report",
        "reports",
        "case",
        "cases",
        "series",
        "data",
        "based",
        "related",
        "associated",
        "compared",
        "comparison",
        "following",
        "during",
        "new",
        "novel",
        "updated",
        "current",
        "recent",
        "retrospective",
        "prospective",
        "multicenter",
        "multicentre",
        "randomized",
        "randomised",
        "controlled",
        "double",
        "blind",
        "single",
        "two",
        "three",
        "first",
        "second",
        "phase",
        "long",
        "short",
        "term",
        "year",
        "years",
        "month",
        "months",
        "day",
        "days",
        "week",
        "weeks",
        "hour",
        "hours",
        "outcome",
        "outcomes",
        "result",
        "results",
        "finding",
        "findings",
        "patient",
        "patients",
        "subject",
        "subjects",
        "adult",
        "adults",
        "population",
        "cohort",
        "group",
        "groups",
        "arm",
        "arms",
        "clinical",
        "medical",
        "surgical",
        "therapeutic",
        "diagnostic",
        "primary",
        "secondary",
        "overall",
        "total",
        "general",
        "high",
        "low",
        "increased",
        "decreased",
        "improved",
        "real",
        "world",
        "large",
        "small",
        "major",
        "minor",
        "initial",
        "final",
        "standard",
        "alternative",
        "six",
        "twelve",
        "twenty",
        "one",
        "ten",
        "hundred",
    }
)

# Termes à faible priorité de tri (non exclus, mais relégués) ───────
# Ces tokens passent le filtre d'exclusion mais ont un IDF bas dans PubMed.
# Ils ne sont PAS retirés de la requête — ils sont triés après les termes
# spécifiques grâce à _term_specificity_score().

_GENERIC_RESEARCH_TERMS: frozenset[str] = frozenset(
    {
        # 1. QUALIFICATIFS PROMOTIONNELS & TEMPORELS (Le "Fluff")
        "rapid", "simple", "quick", "fast", "improved", "enhanced", "effective", 
        "efficient", "optimal", "routine", "novel", "new", "successful", "robust", 
        "comprehensive", "preliminary", "advanced", "basic", "standard", "traditional", 
        "modern", "complex", "current", "recent", "updated",

        # 2. CONTEXTE & LIEUX (Générique)
        "conditions", "condition", "setting", "settings", "field", "context", 
        "region", "center", "centre", "hospital", "clinic", "department", 
        "experience", "practice", "practices", "national", "international", "global",

        # 3. ACTIONS & ÉPISTÉMOLOGIE (Les mots-tiroirs de la recherche)
        "study", "studies", "analysis", "evaluation", "assessment", "investigation", 
        "review", "report", "reports", "comparison", "survey", "observation", 
        "observations", "finding", "findings", "impact", "effect", "effects", 
        "role", "roles", "value", "status", "characterization", "description", 
        "exploration", "association", "relationship",

        # 4. MÉTHODOLOGIE GÉNÉRIQUE (Pour forcer le focus sur la VRAIE méthode)
        "method", "methods", "approach", "approaches", "technique", "techniques", 
        "strategy", "strategies", "procedure", "procedures", "protocol", "protocols", 
        "framework", "concept", "concepts", "design", "development", "validation", 
        "performance", "accuracy", "reliability", "validity", "feasibility", 
        "utility", "quality", "system", "systems", "tool", "tools", "score", 
        "scores", "scale", "scales", "questionnaire", "questionnaires", "index", "indices"
    }
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _term_specificity_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _term_specificity_score(token: str) -> float:
    """
    Score de spécificité d'un token pour le tri des termes de requête.

    Remplace le tri naïf par longueur par un classement en 3 paliers :

      Palier 3.x — Acronyme entièrement en majuscules (SCAN, DREPATEST, AML)
                   Ces tokens identifient presque toujours une entité biomédicale
                   précise : molécule, essai clinique, pathogène, score clinique.

      Palier 2.x — Terme absent de _GENERIC_RESEARCH_TERMS
                   Token spécifique qui a survécu aux stopwords et n'est pas
                   ubiquitaire dans PubMed. Ex : "sickle", "drepatest", "falciparum".

      Palier 1.x — Terme présent dans _GENERIC_RESEARCH_TERMS
                   Token fréquent (IDF bas). Utilisé en dernier recours dans
                   la cascade, pas comme moteur des niveaux 3-5.
                   Ex : "conditions", "accuracy", "settings", "disease".

    Bris d'égalité intra-palier : len(token) / 1000
    Division par 1000 pour que la longueur ne permette
    jamais à un terme générique long de dépasser un acronyme court.

    Clé de tri décroissant : higher → more specific → priorité haute.
    """
    is_acronym: bool = bool(re.match(r"^[A-Z]{2,}(?:[\-/][A-Z0-9]+)*$", token))
    is_generic: bool = token.lower() in _GENERIC_RESEARCH_TERMS

    if is_acronym:
        base: float = 3.0
    elif not is_generic:
        base = 2.0
    else:
        base = 1.0

    return base + len(token) / 1000.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _extract_pubmed_query  (modifié OPT-1 : retourne tokens NON triés)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_pubmed_query(title: str, min_terms: int = 2) -> list[str]:
    """
    Extrait les tokens médicaux significatifs d'un titre d'article.

    Retourne une liste NON TRIÉE — le tri par spécificité est délégué à
    _term_specificity_score() dans _build_cascade_queries() [OPT-1].

    Règles d'inclusion :
      1. Acronymes médicaux (≥2 majuscules, éventuellement séparés par - ou /)
      2. Mots hybrides tiret/slash de longueur ≥ 4
      3. Termes alphabétiques ≥ 5 caractères, hors _MEDICAL_STOPWORDS
      4. Nombres purs : exclus
      5. Fallback si < min_terms retenus : tous tokens hors stopwords
    """
    title = re.sub(r"[®©™°†‡§¶]", "", title)
    tokens = re.findall(r"[\w]+(?:[\-/][\w]+)*", title)

    kept: list[str] = []
    for tok in tokens:
        if re.match(r"^[A-Z]{2,}(?:[\-/][A-Z0-9]{1,})*$", tok):
            kept.append(tok)
            continue
        if ("-" in tok or "/" in tok) and len(tok) >= 4:
            kept.append(tok)
            continue
        lower = tok.lower()
        if lower.isalpha() and len(tok) >= 5 and lower not in _MEDICAL_STOPWORDS:
            kept.append(tok)
            continue

    if len(kept) < min_terms:
        kept = [
            t for t in tokens if t.lower() not in _MEDICAL_STOPWORDS and not t.isdigit()
        ]
    return kept


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [OPT-3] Seuils d'arrêt adaptatifs par niveau de cascade
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Seuil legacy (utilisé comme valeur par défaut dans les utilitaires)
CASCADE_MIN_RESULTS: int = 30

# [OPT-3] Seuils d'arrêt par niveau.
#
# Principe : la qualité des résultats diminue avec le niveau de cascade.
# Un article au niveau 1 (titre exact) est quasi-certain d'être dans la bonne
# discipline. Un article au niveau 5 (OR de termes) est potentiellement du bruit.
# On exige donc MOINS de volume aux niveaux élevés (haute précision) et
# DAVANTAGE de volume aux niveaux bas (faible précision) pour que le signal
# cosinus domine le bruit lors de l'agrégation par journal.
#
# Valeurs calibrées empiriquement :
#   Niveau 1 → 10  : titre exact — 10 articles suffisent, très forte précision
#   Niveau 2 → 20  : termes clés AND — bonne précision, volume modéré ok
#   Niveau 3 → 30  : MeSH canoniques — bonne précision mais besoin de volume
#   Niveau 4 → 50  : AND 2 termes libres — précision moyenne, volume nécessaire
#   Niveau 5 → 70  : OR de termes — faible précision, grand volume requis
#   Niveau 6 → 200 : terme unique — dernier recours, maximiser le volume

_CASCADE_STOP_THRESHOLDS: dict[int, int] = {
    1: 10,
    2: 20,
    3: 30,
    4: 50,
    5: 70,
    6: 200,
}

CASCADE_LEVEL_LABELS: dict[int, tuple[str, str, str]] = {
    0: ("🎯", "#7c3aed", "Requête ancrée sur le domaine détecté (domain_inference)"),
    1: ("✅", "#059669", "Titre complet reconnu par PubMed"),
    2: ("🔵", "#2563eb", "Requête relaxée — termes médicaux clés"),
    3: ("🧬", "#7c3aed", "Termes MeSH canoniques — traduction PubMed"),
    4: ("⚡", "#d97706", "Tous champs PubMed — 2 termes clés (AND)"),
    5: ("🔁", "#d97706", "Tous champs PubMed — termes élargis (OR)"),
    6: ("⚠️", "#dc2626", "Requête minimale — terme le plus spécifique"),
}

# [v8.2] Facteur de surpondération thématique.
# weighted_score += sim × (1 + domain_boost × DOMAIN_BOOST_FACTOR)
# où domain_boost = cos(centroïde_domaine, vecteur_journal) ∈ [0, 1].
# Valeur 0.40 : le domaine peut augmenter le score d'un journal aligné de 40%
# sans écraser le signal cosinus titre↔article pour les journaux très pertinents.
DOMAIN_BOOST_FACTOR: float = 0.40


# ── Pipeline MeSH TF-IDF + rerankage cosinus ──────────────────────────────────


def _fetch_mesh_translations(title: str) -> dict[str, int]:
    """Interroge PubMed esearch, retourne {terme_mesh: TF}. {} en cas d'échec."""
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        resp = requests.get(
            f"{BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": title,
                "retmax": 0,
                "retmode": "json",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("esearchresult", {})
        tf_counts: dict[str, int] = {}
        for translation in data.get("translationset", []):
            raw: str = translation.get("to", "")
            for term in re.findall(r'"([^"]+)"\[MeSH Terms\]', raw, re.IGNORECASE):
                key = term.lower()
                tf_counts[key] = tf_counts.get(key, 0) + 1
        logger.info("MeSH TF pour %r : %s", title[:50], tf_counts)
        return tf_counts
    except (requests.RequestException, Exception) as e:
        logger.warning("_fetch_mesh_translations: %s", e)
        return {}


_PUBMED_TOTAL_DOCS: int = 35_000_000
_MESH_IDF_CACHE: dict[str, int] = {}


def _get_mesh_doc_count(term: str) -> tuple[str, int]:
    """Retourne (term, doc_count) depuis le cache ou PubMed. Fallback → _PUBMED_TOTAL_DOCS."""
    if term in _MESH_IDF_CACHE:
        return term, _MESH_IDF_CACHE[term]
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        resp = requests.get(
            f"{BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f'"{term}"[MeSH Terms]',
                "retmax": 0,
                "retmode": "json",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        count = int(
            resp.json().get("esearchresult", {}).get("count", _PUBMED_TOTAL_DOCS)
        )
        result = max(count, 1)
        _MESH_IDF_CACHE[term] = result
        return term, result
    except Exception as e:
        logger.warning("_get_mesh_doc_count(%r): %s", term, e)
        return term, _PUBMED_TOTAL_DOCS


def _score_mesh_terms(
    tf_counts: dict[str, int], top_k: int = 8, min_tfidf: float = 3.0
) -> list[str]:
    """
    TF-IDF des termes MeSH. top_k=8 par défaut pour élargir le pool
    avant rerankage cosinus. Requêtes IDF parallèles via ThreadPoolExecutor.
    """
    if not tf_counts:
        return []
    with ThreadPoolExecutor(max_workers=min(len(tf_counts), 8)) as exe:
        idf_results: list[tuple[str, int]] = list(
            exe.map(_get_mesh_doc_count, tf_counts.keys())
        )
    scored: list[tuple[float, str]] = []
    for term, doc_count in idf_results:
        tf: int = tf_counts.get(term, 1)
        idf: float = math.log(_PUBMED_TOTAL_DOCS / doc_count)
        tfidf: float = tf * idf
        logger.info(
            "MeSH TF-IDF: %r → TF=%d, doc_count=%d, IDF=%.2f, score=%.2f",
            term,
            tf,
            doc_count,
            idf,
            tfidf,
        )
        if tfidf >= min_tfidf:
            scored.append((tfidf, term))
    scored.sort(reverse=True)
    return [term for _, term in scored[:top_k]]


# APRÈS
def _rerank_mesh_by_cosine(
    title: str,
    mesh_terms: list[str],
    top_k: int = 3,
    min_cosine: float = 0.20,  # ← seuil de pertinence minimale
) -> list[str]:
    """
    Rerankage cosinus titre ↔ termes MeSH.

    min_cosine : seuil de rejet. Un terme MeSH dont la similarité cosinus
    avec le titre est inférieure à ce seuil est considéré hors-domaine et
    écarté, même s'il figurait dans le top TF-IDF.
    Valeur 0.20 empiriquement robuste sur multilingual-e5-small :
      - "home environment" vs titre chimio → ~0.08-0.12 (rejeté ✅)
      - "anemia, sickle cell" vs titre drépanocytose → ~0.35-0.55 (gardé ✅)
    """
    if not mesh_terms:
        return []
    if _EMBED_MODEL is None:
        logger.debug("_rerank_mesh_by_cosine : pas de modèle, fallback TF-IDF order")
        return mesh_terms[:top_k]

    title_emb: np.ndarray = _e5_encode(
        _EMBED_MODEL, [title], prefix="query", normalize=True
    )[0]
    term_embs: np.ndarray = _e5_encode(
        _EMBED_MODEL, mesh_terms, prefix="passage", normalize=True
    )
    cosine_scores: np.ndarray = term_embs @ title_emb

    ranked_indices: list[int] = list(np.argsort(-cosine_scores))

    # Filtre : on rejette les termes sous le seuil de pertinence
    filtered: list[str] = [
        mesh_terms[i] for i in ranked_indices if cosine_scores[i] >= min_cosine
    ]

    logger.info(
        "_rerank_mesh_by_cosine : %s (seuil=%.2f, rejetés=%d)",
        [(mesh_terms[i], round(float(cosine_scores[i]), 3)) for i in ranked_indices],
        min_cosine,
        len(mesh_terms) - len(filtered),
    )
    return filtered[:top_k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [OPT-2] _mesh_term_to_plain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mesh_term_to_plain(mesh_term: str) -> str:
    """
    [OPT-2] Convertit un terme MeSH canonique en phrase de recherche libre.

    Les termes MeSH utilisent la convention "concept général, qualificatif"
    inversée par rapport à l'anglais naturel.
    Cette fonction réordonne pour maximiser le recall en texte libre.

    Exemples :
        "anemia, sickle cell"  → "sickle cell anemia"
        "africa, western"      → "western africa"
        "hemoglobin sc disease"→ "hemoglobin sc disease"  (pas de virgule → inchangé)

    La réorganisation autour de la virgule est une convention MeSH fiable :
    tout ce qui est après la virgule est le qualificatif/modificateur.
    """
    parts = [p.strip() for p in mesh_term.split(",", maxsplit=1)]
    if len(parts) == 2 and parts[1]:
        # Réordonne : "qualificatif concept" → "qualificatif concept"
        return f"{parts[1]} {parts[0]}"
    return mesh_term


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _build_cascade_queries  (modifié OPT-1 + OPT-2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_cascade_queries(title: str) -> list[tuple[str, str]]:
    """
    Cascade de 6 requêtes (term, field_tag) de plus en plus relaxées.

    Niveau 1 : titre complet (® nettoyé)           + [Title/Abstract]
    Niveau 2 : termes médicaux clés (AND)           + [Title/Abstract]
    Niveau 3 : termes MeSH TF-IDF rerankés cosinus  + [MeSH Terms]   ← pipeline MeSH
    Niveau 4 : [OPT-2] 2 termes MeSH plain text OR 2 tokens spécifiques
    Niveau 5 : [OPT-2] 3-5 termes MeSH plain text OR 5 tokens spécifiques
    Niveau 6 : terme le plus spécifique seul (token ou MeSH plain)

    [OPT-1] Le tri kw_sorted utilise _term_specificity_score() au lieu de len().
             Résultat : les acronymes (SCAN, DREPATEST) et termes rares (sickle)
             passent devant les termes génériques (conditions, accuracy, settings).

    [OPT-2] Quand best_mesh est disponible, les niveaux 4/5/6 sont dérivés des
             termes MeSH canoniques convertis en texte libre par _mesh_term_to_plain().
             Ex : best_mesh = ["anemia, sickle cell", "africa, western"]
                  → mesh_plain = ["sickle cell anemia", "western africa"]
                  → niveau 4 : "sickle cell anemia OR western africa"  (au lieu de "settings DREPATEST")
             Fallback vers les tokens bruts si best_mesh est vide.
    """
    # Nettoyage des caractères spéciaux (® casse les requêtes PubMed)
    clean_title: str = re.sub(r"[®©™°†‡§¶]", "", title).strip()

    # Extraction des tokens (liste non triée)
    keywords: list[str] = _extract_pubmed_query(clean_title)

    # [OPT-1] Tri par spécificité médicale réelle (score décroissant)
    kw_sorted: list[str] = sorted(keywords, key=_term_specificity_score, reverse=True)

    # ── Pipeline MeSH (TF-IDF → rerankage cosinus) ────────────────────────────
    tf_counts: dict[str, int] = _fetch_mesh_translations(clean_title)
    tfidf_candidates: list[str] = _score_mesh_terms(tf_counts, top_k=8, min_tfidf=3.0)
    best_mesh: list[str] = _rerank_mesh_by_cosine(
        clean_title, tfidf_candidates, top_k=3
    )

    # ── [OPT-2] Termes MeSH convertis en texte libre ──────────────────────────
    # _mesh_term_to_plain réordonne "concept, qualificatif" → "qualificatif concept"
    # pour maximiser le recall en recherche texte libre aux niveaux 4/5/6.
    mesh_plain: list[str] = [_mesh_term_to_plain(t) for t in best_mesh]

    # ── Construction des requêtes par niveau ──────────────────────────────────

    # Niveau 1 : titre complet nettoyé
    q1_term: str = clean_title

    # Niveau 2 : AND implicite de tous les tokens (max 6 pour éviter 0 résultat)
    q2_term: str = " ".join(kw_sorted[:6])

    # Niveau 3 : requête MeSH structurée (field tag = "" car les [MeSH Terms] sont
    # déjà inclus dans q2b_term via la syntaxe "terme"[MeSH Terms])
    q3_term: str = (
        " AND ".join(f'"{t}"[MeSH Terms]' for t in best_mesh) if best_mesh else ""
    )

    # Niveau 4 : [OPT-2] 2 termes MeSH plain text si disponibles, sinon 2 tokens spécifiques
    if len(mesh_plain) >= 2:
        q4_term: str = " AND ".join(mesh_plain[:2])
    elif len(mesh_plain) == 1:
        q4_term = mesh_plain[0]
    else:
        # Fallback tokens bruts — kw_sorted[:2] favorise maintenant les acronymes
        q4_term = (
            " ".join(kw_sorted[:2])
            if len(kw_sorted) >= 2
            else (kw_sorted[0] if kw_sorted else "")
        )

    # Niveau 5 : [OPT-2] tous les termes MeSH plain text en OR (jusqu'à 3) si disponibles,
    # sinon 5 tokens spécifiques en OR
    if mesh_plain:
        q5_term: str = " OR ".join(mesh_plain[:3])
    else:
        q5_term = " OR ".join(kw_sorted[:5]) if kw_sorted else ""

    # Niveau 6 : terme unique le plus spécifique
    # [OPT-2] Priorité au premier terme MeSH plain (déjà désambigüisé par PubMed),
    # sinon au premier token de kw_sorted (acronyme ou terme rare en tête grâce à OPT-1)
    if mesh_plain:
        q6_term: str = mesh_plain[0]
    else:
        q6_term = kw_sorted[0] if kw_sorted else ""

    raw: list[tuple[str, str]] = [
        (q1_term, "[Title/Abstract]"),
        (q2_term, "[Title/Abstract]"),
        (q3_term, ""),  # MeSH TF-IDF — field tag inclus dans le terme
        (q4_term, ""),
        (q5_term, ""),
        (q6_term, ""),
    ]

    # Déduplication en conservant l'ordre
    seen: set[tuple[str, str]] = set()
    cascade: list[tuple[str, str]] = []
    for term, field in raw:
        term = term.strip()
        if term and (term, field) not in seen:
            seen.add((term, field))
            cascade.append((term, field))
    return cascade


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# suggest_journals_by_title 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def suggest_journals_by_title(
    article_title: str,
    ref_db: Optional[SigapsRefDB] = None,
    max_articles: int = 100,
    max_journals: int = 10,
    domain_engine=None,  # Optional[HierarchicalDomainInference] — None → comportement v8.1
) -> list[SuggestedJournal]:
    """
    Surpondération thématique via domain_inference.py + enrichissement Scopus.

    domain_engine != None :
      ① Niveau 0 de cascade : requête PubMed ancrée sur les descripteurs canoniques
         du domaine détecté — arrive AVANT les 6 niveaux existants.
      ② DOMAIN_BOOST_FACTOR : weighted_score(journal_j) += sim × (1 + boost_j × 0.40)
         où boost_j = cos(centroïde_domaine, vecteur_journal_j).
         Les journaux sémantiquement alignés avec le domaine sont surpondérés sans
         écraser le signal cosinus titre↔article pour les journaux très pertinents.
      ③ Scopus thématique : ScopusFetcher.search_by_topic() enrichit le pool d'articles
         avec des publications alignées sur les termes du domaine détecté.

    domain_engine=None → aucune régression
    """
    db = ref_db or _REF_DB
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # ── Détection du domaine sémantique ───────────────────────────────────
    # Calculs effectués une seule fois ici — detect_article_domain() appelle
    # domain_engine.predict() séparément dans app.py (cache session_state).
    _domain_centroid: Optional[np.ndarray] = None
    _domain_level0_terms: list[str] = []

    if domain_engine is not None:
        try:
            _report = domain_engine.predict(article_title)
            if not _report.is_out_of_scope and _report.best:
                _best_domain = _report.best.domain
                # Centroïde : spécialiste si routé, général sinon
                if _report.routed_to_specialist:
                    _domain_centroid = domain_engine._oncology_centroids.get(_best_domain.id)
                else:
                    _domain_centroid = domain_engine._general_centroids.get(_best_domain.id)
                # Termes niveau 0 : tokens de la description canonique du domaine
                if _best_domain.descriptions:
                    _desc_tokens = _extract_pubmed_query(_best_domain.descriptions[0])
                    _domain_level0_terms = sorted(
                        _desc_tokens, key=_term_specificity_score, reverse=True
                    )[:4]
                logger.info(
                    "suggest_journals domain_inference : domaine=%r conf=%.2f termes=%r",
                    _best_domain.id,
                    _report.best.confidence,
                    _domain_level0_terms,
                )
        except Exception as _exc:
            logger.warning("suggest_journals domain_engine.predict(): %s — ignoré", _exc)

    # ── Construction de la cascade (niveau 0 si domaine détecté) ─────────────
    _base_cascade = _build_cascade_queries(article_title)
    cascade_queries: list[tuple[str, str]] = []
    if _domain_level0_terms:
        _lvl0_term = " AND ".join(_domain_level0_terms)
        cascade_queries = [(_lvl0_term, "[Title/Abstract]")] + _base_cascade
        _level_offset = 0   # enumerate démarre à 0
    else:
        cascade_queries = _base_cascade
        _level_offset = 1   # enumerate démarre à 1

    uids: list[str] = []
    query_used: str = ""
    query_level: int = 0

    for level, (term, field_tag) in enumerate(cascade_queries, start=_level_offset):
        pubmed_term = f"{term}{field_tag}" if field_tag else term
        try:
            s_resp = requests.get(
                f"{BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": pubmed_term,
                    "retmax": max_articles,
                    "retmode": "json",
                    "sort": "relevance",
                    **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
                },
                timeout=REQUEST_TIMEOUT,
            )
            s_resp.raise_for_status()
            candidate_uids = s_resp.json().get("esearchresult", {}).get("idlist", [])
        except requests.RequestException as e:
            logger.warning("suggest_journals esearch level %d: %s", level, e)
            continue

        if candidate_uids:
            if len(candidate_uids) > len(uids):
                uids = candidate_uids
                query_used = term
                query_level = level

            logger.info(
                "suggest_journals : %d résultats au niveau %d (terme: %r, field: %r)",
                len(candidate_uids),
                level,
                term,
                field_tag,
            )

            # Seuil adaptatif : niveau 0 = même exigence que niveau 1
            _effective_level = max(level, 1)
            stop_threshold: int = _CASCADE_STOP_THRESHOLDS.get(
                _effective_level, CASCADE_MIN_RESULTS
            )
            if len(uids) >= stop_threshold:
                break

    if not uids:
        return []

    # ── Récupération détails XML ──────────────────────────────────────────────
    try:
        f_resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(uids),
                "retmode": "xml",
                **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}),
            },
            timeout=REQUEST_TIMEOUT,
        )
        f_resp.raise_for_status()
        root = ET.fromstring(f_resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("suggest_journals efetch: %s", e)
        return []

    # ── Collecte métadonnées ──────────────────────────────────────────────────
    article_meta: list[dict] = []
    for art_tag in root.findall(".//PubmedArticle"):
        nlm_tag = art_tag.find(".//MedlineJournalInfo/NlmUniqueID")
        title_tag = art_tag.find(".//ArticleTitle")
        if nlm_tag is None or title_tag is None:
            continue
        art_title = html.unescape(
            ET.tostring(title_tag, method="text", encoding="unicode")
        ).strip()
        jrnl_tag = art_tag.find(".//Journal/Title")
        mta_tag = art_tag.find(".//MedlineJournalInfo/MedlineTA")
        issn_tag = art_tag.find(".//ISSN")
        pmid_tag = art_tag.find(".//MedlineCitation/PMID")
        art_pmid = (pmid_tag.text or "").strip() if pmid_tag is not None else ""
        article_meta.append(
            {
                "nlm_id": (nlm_tag.text or "").strip(),
                "art_title": art_title,
                "art_pmid": art_pmid,
                "jrnl_name": jrnl_tag.text if jrnl_tag is not None else "",
                "medline_ta": mta_tag.text if mta_tag is not None else "—",
                "issn": issn_tag.text if issn_tag is not None else "",
            }
        )

    if not article_meta:
        return []

    # ── Calcul des similarités ────────────────────────────────────────────────
    model = _EMBED_MODEL
    use_transformer = model is not None

    if use_transformer:
        all_texts = [article_title] + [m["art_title"] for m in article_meta]
        all_embs = _e5_encode(model, all_texts, prefix="query", normalize=True)
        query_vec_s = all_embs[0]
        art_vecs = all_embs[1:]
        sims = (art_vecs @ query_vec_s).tolist()
    else:
        query_tri = _trigrams(article_title)
        sims = [_jaccard(query_tri, _trigrams(m["art_title"])) for m in article_meta]

    # ── Pré-calcul index NLM→idx pour le boost domaine (O(1) par article) ──
    _nlm_to_idx: dict[str, int] = {}
    if _domain_centroid is not None and db.embeddings_ready and db._emb_nlm_ids:
        _nlm_to_idx = {nlm: i for i, nlm in enumerate(db._emb_nlm_ids)}

    # ── Agrégation par journal ────────────────────────────────────────────────
    journal_agg: dict[str, dict] = {}
    for meta, sim in zip(article_meta, sims):
        nlm_id = meta["nlm_id"]
        if nlm_id not in journal_agg:
            journal_agg[nlm_id] = {
                "weighted_score": 0.0,
                "count": 0,
                "journal_name": meta["jrnl_name"],
                "medline_ta": meta["medline_ta"],
                "issn": meta["issn"],
                "example_titles": [],
            }

        # Boost domaine : cos(centroïde_domaine, vecteur_journal) × DOMAIN_BOOST_FACTOR
        _boost: float = 0.0
        if _domain_centroid is not None and nlm_id in _nlm_to_idx:
            _j_vec: np.ndarray = db._emb_matrix[_nlm_to_idx[nlm_id]]
            _boost = max(float(np.dot(_domain_centroid, _j_vec)), 0.0) * DOMAIN_BOOST_FACTOR

        journal_agg[nlm_id]["weighted_score"] += sim * (1.0 + _boost)
        journal_agg[nlm_id]["count"] += 1
        if len(journal_agg[nlm_id]["example_titles"]) < 3:
            journal_agg[nlm_id]["example_titles"].append(
                {"title": meta["art_title"], "pmid": meta["art_pmid"]}
            )

    # ── Enrichissement Scopus thématique ───────────────────────────────
    # Activé uniquement si le domaine a été détecté (termes niveau 0 disponibles).
    # Ne touche pas à la recherche auteur/équipe (FederatedSearch.search()).
    if _domain_level0_terms:
        _scopus = ScopusFetcher(ref_db=db)
        _scopus_arts: list[dict] = _scopus.search_by_topic(
            _domain_level0_terms, limit=50
        )
        if _scopus_arts and model is not None:
            _sc_art_titles = [a["art_title"] for a in _scopus_arts if a["art_title"]]
            if _sc_art_titles:
                _sc_vecs: np.ndarray = _e5_encode(
                    model, _sc_art_titles, prefix="query", normalize=True
                )
                _sc_sims: list[float] = (_sc_vecs @ query_vec_s).tolist()

                for sc_art, sc_sim in zip(
                    [a for a in _scopus_arts if a["art_title"]], _sc_sims
                ):
                    sc_journal: str = sc_art["journal_name"]
                    # Résolution NLM_ID via fuzzy search dans le référentiel CSV
                    _sc_matches = db.fuzzy_search(sc_journal, top_k=1, threshold=0.15)
                    if _sc_matches:
                        sc_nlm_id: str = _sc_matches[0]["nlm_id"]
                    else:
                        # Clé de repli : ISSN ou nom tronqué
                        sc_nlm_id = sc_art.get("issn") or f"SCOPUS:{sc_journal[:24]}"

                    if not sc_nlm_id:
                        continue

                    _sc_boost: float = 0.0
                    if _domain_centroid is not None and sc_nlm_id in _nlm_to_idx:
                        _j_vec = db._emb_matrix[_nlm_to_idx[sc_nlm_id]]
                        _sc_boost = (
                            max(float(np.dot(_domain_centroid, _j_vec)), 0.0)
                            * DOMAIN_BOOST_FACTOR
                        )

                    if sc_nlm_id not in journal_agg:
                        journal_agg[sc_nlm_id] = {
                            "weighted_score": 0.0,
                            "count": 0,
                            "journal_name": sc_journal,
                            "medline_ta": "—",
                            "issn": sc_art.get("issn", ""),
                            "example_titles": [],
                        }
                    journal_agg[sc_nlm_id]["weighted_score"] += sc_sim * (1.0 + _sc_boost)
                    journal_agg[sc_nlm_id]["count"] += 1
                    if (
                        len(journal_agg[sc_nlm_id]["example_titles"]) < 3
                        and sc_art["art_title"]
                    ):
                        journal_agg[sc_nlm_id]["example_titles"].append(
                            {"title": sc_art["art_title"], "pmid": ""}
                        )

    if not journal_agg:
        return []

    max_score = max(v["weighted_score"] for v in journal_agg.values()) or 1.0

    # ── Jointure CSV + construction SuggestedJournal ──────────────────────────
    suggestions: list[SuggestedJournal] = []
    for nlm_id, agg in journal_agg.items():
        rank, rank_source = _resolve_rank(agg["journal_name"], nlm_id, db)
        impact_factor = db.get_if(nlm_id) if db.loaded else ""
        medline_indexed = db.get_medline(nlm_id) if db.loaded else ""
        normalized_sim = round(agg["weighted_score"] / max_score, 4)
        suggestions.append(
            SuggestedJournal(
                nlm_id=nlm_id,
                journal_name=agg["journal_name"] or nlm_id,
                medline_ta=agg["medline_ta"],
                rank=rank,
                rank_source=rank_source,
                impact_factor=impact_factor,
                medline_indexed=medline_indexed,
                issn=agg["issn"],
                country="",
                similarity=normalized_sim,
                article_count=agg["count"],
                example_titles=agg["example_titles"],
                query_level=query_level,
                query_used=query_used,
            )
        )

    suggestions.sort(key=lambda x: -x.similarity)
    return suggestions[:max_journals]