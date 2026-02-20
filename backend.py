"""
backend.py — OpenSIGAPS v7
Moteur PubMed (OpenAlex supprimé). Algorithme SIGAPS conforme MERRI 2022.

NOUVEAUTÉS v7 :
  • OpenAlex supprimé — PubMed couvre déjà l'intégralité des publications
  • SigapsRefDB : stocke aussi l'Impact Factor (colonne *_IF la plus récente)
  • fuzzy_search_csv() : similarité Jaccard sur trigrammes, O(n) pur Python,
    < 80 ms sur 25 000 lignes, zéro dépendance externe
  • search_journal_by_name() : double moteur API NLM Catalog + fuzzy CSV,
    fusion et dédup par NLM_ID, tri par score de similarité

Référence : DRCI AP-HP — MERRI 2022
https://recherche-innovation.aphp.fr/wp-content/blogs.dir/77/files/2022/04/
SIGAPS-Nouveautes-diffusion-V2.pdf
"""

import csv
import html
import logging
import pickle
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ── Disponibilité des backends d'embedding ───────────────────────────────────
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

# Répertoire ONNX (à côté d'app.py par convention)
ONNX_MODEL_DIR = "multilingual-e5-small-onnx"


class _ONNXEmbedder:
    """
    Wrapper léger autour de ORTModelForFeatureExtraction.

    Expose la même interface que SentenceTransformer (.encode()) pour que
    _e5_encode() reste inchangé. Effectue le mean-pooling + normalisation L2
    directement avec numpy (pas de dépendance PyTorch à l'inférence).

    Chargement : ~400 ms (vs ~3-4 s PyTorch).
    Inférence 1 phrase : ~8 ms (vs ~30 ms PyTorch).
    RAM : ~90 MB quantifié int8 (vs ~500 MB PyTorch).
    """

    def __init__(self, model_dir: str) -> None:
        self._model = ORTModelForFeatureExtraction.from_pretrained(model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> "np.ndarray":
        all_parts: list[np.ndarray] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = self._model(**encoded)

            # Mean pooling pondéré par le masque d'attention
            token_emb = outputs.last_hidden_state.detach().numpy()  # (B, L, D)
            attn_mask = encoded["attention_mask"].numpy()  # (B, L)
            mask_exp = attn_mask[:, :, np.newaxis].astype(np.float32)
            sum_emb = (token_emb * mask_exp).sum(axis=1)  # (B, D)
            sum_mask = mask_exp.sum(axis=1).clip(1e-9, None)  # (B, 1)
            embeddings = (sum_emb / sum_mask).astype(np.float32)  # (B, D)

            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(
                    1e-9, None
                )
                embeddings = embeddings / norms

            all_parts.append(embeddings)

        return np.vstack(all_parts)


def load_embed_model(onnx_dir: str = ONNX_MODEL_DIR):
    """
    Charge le modèle d'embedding avec priorité ONNX > PyTorch > None.

    1. Si le répertoire ONNX existe et optimum est installé → _ONNXEmbedder
    2. Sinon si sentence-transformers est installé → SentenceTransformer
    3. Sinon → None (fallback Jaccard)

    Retourne un tuple (model_or_None, backend_str).
    backend_str ∈ {"onnx", "pytorch", "jaccard"}
    """
    if _ONNX_AVAILABLE and Path(onnx_dir).exists():
        try:
            model = _ONNXEmbedder(onnx_dir)
            logger.info("Modèle chargé via ONNX Runtime depuis %s", onnx_dir)
            return model, "onnx"
        except Exception as e:
            logger.warning("Échec chargement ONNX (%s), tentative PyTorch", e)

    if _ST_AVAILABLE:
        try:
            model = SentenceTransformer("intfloat/multilingual-e5-small")
            logger.info("Modèle chargé via sentence-transformers (PyTorch)")
            return model, "pytorch"
        except Exception as e:
            logger.warning("Échec chargement PyTorch : %s", e)

    logger.warning("Aucun modèle d'embedding disponible — fallback Jaccard")
    return None, "jaccard"


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

VALEUR_POINT_EUROS = 650
REQUEST_TIMEOUT = 10

# ─────────────────────────────────────────────
# COEFFICIENTS OFFICIELS MERRI 2022
# ─────────────────────────────────────────────

C1_COEFFICIENTS: dict[str, int] = {
    "1er": 4,
    "2ème": 3,
    "3ème": 2,
    "ADA": 3,  # Avant-dernier — UNIQUEMENT si nb_auteurs >= 6
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES FUZZY — trigrammes Jaccard, zéro dépendance
# ═══════════════════════════════════════════════════════════════════════════════


def _trigrams(s: str) -> set[str]:
    """Génère l'ensemble des trigrammes de caractères d'une chaîne normalisée."""
    s = " " + re.sub(r"[^a-z0-9 ]", " ", s.lower()) + " "
    s = re.sub(r"\s+", " ", s)
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGAPS REFERENCE DATABASE
# ═══════════════════════════════════════════════════════════════════════════════


class SigapsRefDB:
    """
    Lit sigaps_ref.csv avec csv.DictReader et construit en un seul passage :

      _rank_by_nlm    : dict[nlm_id → rank]
      _if_by_nlm      : dict[nlm_id → impact_factor_str]
      _info_by_nlm    : dict[nlm_id → {journal, rank, if, trigrams}]

    Les trigrammes sont pré-calculés au chargement (< 50 ms sur 25k lignes)
    pour que fuzzy_search() soit O(n) sans recalcul.

    Colonnes CSV minimales : NLM_ID, Journal, Latest_Rank
    Colonne IF optionnelle : toute colonne dont le nom contient « _IF »
                             (ex: 2020_IF, 2022_IF) — la plus récente est utilisée.
    """

    VALID_RANKS = {"A+", "A", "B", "C", "D", "E", "NC"}

    def __init__(self) -> None:
        self._rank_by_nlm: dict[str, str] = {}
        self._if_by_nlm: dict[str, str] = {}
        self._medline_by_nlm: dict[str, str] = {}  # 'Oui' | 'Non' | ''
        self._info_by_nlm: dict[str, dict] = {}
        self._history_by_nlm: dict[str, dict] = {}  # {nlm_id → {year → {rank, if}}}
        self._if_cols_years: list[tuple] = []  # [(year, col_name), ...] pour _ingest
        self.loaded = False
        self.source_path = ""
        self.row_count = 0
        # ── Embeddings ──────────────────────────────────────────────────
        # _emb_matrix : float32 normalisé L2, shape (N, 384)
        # _emb_nlm_ids : list[str] — nlm_id[i] correspond à _emb_matrix[i]
        self._emb_matrix: Optional["np.ndarray"] = None
        self._emb_nlm_ids: list[str] = []
        self.embeddings_ready: bool = False

    # ── Chargement ───────────────────────────────────────────────────────────

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

        # Colonnes Impact Factor multi-années : YYYY_IF
        import re as _re

        all_if_cols = sorted(
            [f for f in fields if f.upper().endswith("_IF") or f.upper() == "IF"],
            reverse=True,
        )
        if_col = all_if_cols[0] if all_if_cols else None

        # Parser les années disponibles pour l'historique
        if_year_cols: list[tuple[int, str]] = []
        for col in all_if_cols:
            m = _re.match(r"^(\d{4})_IF$", col, _re.IGNORECASE)
            if m:
                if_year_cols.append((int(m.group(1)), col))

        # Colonnes rang par année : YYYY_Rank ou YYYY_Rang
        rank_year_cols: list[tuple[int, str]] = []
        for f in fields:
            m = _re.match(r"^(\d{4})_Rank$|^(\d{4})_Rang$", f, _re.IGNORECASE)
            if m:
                year_v = int(m.group(1) or m.group(2))
                rank_year_cols.append((year_v, f))

        if not (rank_col and nlm_col):
            logger.error(
                "CSV : colonnes NLM_ID ou rank introuvables. Colonnes : %s", fields
            )
            return

        for row in reader:
            nlm = (row.get(nlm_col) or "").strip()
            rank = (row.get(rank_col) or "").strip().upper()
            jrnl = (row.get(journal_col) or "").strip() if journal_col else ""
            if_v = ""
            if if_col:
                raw = (row.get(if_col) or "").strip()
                # Nettoie les valeurs comme "12,34" → "12.34"
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

            # Historique multi-années
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

    # ── Accès directs ─────────────────────────────────────────────────────────

    def get_rank(self, nlm_id: str) -> Optional[str]:
        return self._rank_by_nlm.get(nlm_id.strip()) if nlm_id else None

    def get_if(self, nlm_id: str) -> str:
        return self._if_by_nlm.get(nlm_id.strip(), "") if nlm_id else ""

    def get_medline(self, nlm_id: str) -> str:
        return self._medline_by_nlm.get(nlm_id.strip(), "") if nlm_id else ""

    def get_info(self, nlm_id: str) -> Optional[dict]:
        return self._info_by_nlm.get(nlm_id.strip()) if nlm_id else None

    def get_history(self, nlm_id: str) -> dict:
        """
        Retourne l'historique {année → {rank, if}} du journal identifié par nlm_id.
        Si le CSV ne contient pas de colonnes YYYY_IF ou YYYY_Rank, retourne {}.
        """
        return self._history_by_nlm.get(nlm_id.strip(), {}) if nlm_id else {}

    # ── Embeddings ────────────────────────────────────────────────────────────

    def build_embeddings(
        self,
        model,
        cache_dir: Optional["str | Path"] = None,
        progress_callback=None,
    ) -> bool:
        """
        Calcule et stocke les embeddings de tous les noms de journaux du CSV.

        Stratégie de cache disque (à côté du CSV) :
          sigaps_ref.emb.npy   — matrice float32 (N, 384), normalisée L2
          sigaps_ref.emb.ids   — liste pickle des NLM_IDs dans l'ordre des lignes

        Si le cache existe et que le nombre de lignes correspond, le chargement
        depuis disque est quasi-instantané (~100 ms). Sinon, on encode et on sauve.

        progress_callback(int) : appelé avec le % d'avancement (0→100) si fourni.
        Retourne True si les embeddings sont disponibles après l'appel.
        """
        if not self.loaded or model is None:
            return False

        # Chemins de cache
        base = Path(cache_dir or self.source_path).with_suffix("")
        path_emb = Path(str(base) + ".emb.npy")
        path_ids = Path(str(base) + ".emb.ids")

        # ── Tentative de chargement depuis le cache ───────────────────────────
        if self._try_load_cache(path_emb, path_ids):
            logger.info(
                "SigapsRefDB : embeddings chargés depuis cache (%d vecteurs)",
                len(self._emb_nlm_ids),
            )
            self.embeddings_ready = True
            if progress_callback:
                progress_callback(100)
            return True

        # ── Calcul des embeddings ─────────────────────────────────────────────
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

        matrix = np.vstack(parts).astype(np.float32)  # (N, 384)

        # ── Sauvegarde cache ──────────────────────────────────────────────────
        try:
            np.save(path_emb, matrix)
            with open(path_ids, "wb") as f:
                pickle.dump(nlm_ids, f)
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
        """Charge le cache si cohérent avec le CSV actuel."""
        if not path_emb.exists() or not path_ids.exists():
            return False
        try:
            matrix = np.load(path_emb)
            with open(path_ids, "rb") as f:
                nlm_ids = pickle.load(f)
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

    def semantic_search(
        self,
        query_vec: "np.ndarray",
        top_k: int = 8,
    ) -> list[dict]:
        """
        Recherche cosine sur la matrice pré-calculée.
        query_vec : float32 normalisé L2, shape (384,)
        Retourne top_k dicts {nlm_id, journal, rank, if, medline, similarity}.

        Complexité O(N·384) — ~0.3 ms sur 25k vecteurs (BLAS).
        """
        if not self.embeddings_ready or self._emb_matrix is None:
            return []

        # Cosine = dot-product (vecteurs déjà normalisés L2)
        scores = self._emb_matrix @ query_vec  # shape (N,)
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

    # ── Recherche floue (conservée comme fallback sans modèle) ────────────────

    # ── Recherche floue ────────────────────────────────────────────────────────

    def fuzzy_search(
        self,
        query: str,
        top_k: int = 8,
        threshold: float = 0.08,
    ) -> list[dict]:
        """
        Similarité Jaccard sur trigrammes entre `query` et tous les journaux du CSV.
        Complexité O(n), ~80 ms sur 25 000 entrées. Zéro dépendance externe.

        Retourne une liste de dicts triée par score décroissant :
          {nlm_id, journal, rank, if, similarity}
        """
        if not self.loaded:
            return []
        q_tri = _trigrams(query)
        if not q_tri:
            return []

        scored: list[tuple[float, str, dict]] = []
        for nlm_id, info in self._info_by_nlm.items():
            sim = _jaccard(q_tri, info["trigrams"])
            if sim >= threshold:
                scored.append((sim, nlm_id, info))

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


# ── Singleton module-level ────────────────────────────────────────────────────

_REF_DB: SigapsRefDB = SigapsRefDB()

# ── Singleton modèle d'embedding ─────────────────────────────────────────────

_EMBED_MODEL = None  # _ONNXEmbedder | SentenceTransformer | None
_EMBED_BACKEND = "jaccard"  # "onnx" | "pytorch" | "jaccard"


def set_embed_model(model, backend: str = "jaccard") -> None:
    """Injecte le modèle et le backend chargés par app.py."""
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
    prefix: str = "query",
    batch_size: int = 256,
    normalize: bool = True,
) -> "np.ndarray":
    """
    Encode une liste de textes avec le préfixe e5 obligatoire.
    prefix ∈ {"query", "passage"}
    Retourne un array float32 normalisé L2, shape (len(texts), dim).
    La normalisation L2 permet cosine = dot-product au moment de la recherche.
    """
    prefixed = [f"{prefix}: {t}" for t in texts]
    embs = model.encode(
        prefixed,
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


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE CALCUL SIGAPS
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_presence_score(position: str, rank: str) -> float:
    return SIGAPS_MATRIX.get(position, {}).get(rank, 0.0)


def calculate_sum_c1(nb_authors: int) -> float:
    """Σ C1 pour le score fractionnaire. ADA actif si nb_authors ≥ 6."""
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
    """Score présence ÉQUIPE : un article par structure, meilleure position."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLE DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════


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
    rank_source: str = "heuristique"  # "csv" | "heuristique"

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


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES PARTAGÉS
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_for_dedup(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\W+", "", text.lower())


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
    journal_name: str,
    nlm_id: str = "",
    ref_db: Optional[SigapsRefDB] = None,
) -> tuple[str, str]:
    """
    Résout rang + source. Priorité : CSV (NLM_ID) > heuristique nom.
    Retourne (rank, source) où source ∈ {"csv", "heuristique"}.
    """
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
    if idx == 1 and n > 1:
        return "2ème"
    if idx == 2 and n > 2:
        return "3ème"
    if idx == n - 1:
        return "Dernier"
    if idx == n - 2 and n >= 6:
        return "ADA"
    return "Autre"


# ═══════════════════════════════════════════════════════════════════════════════
# FETCHER PUBMED (seul moteur — OpenAlex supprimé)
# ═══════════════════════════════════════════════════════════════════════════════


class PubMedFetcher:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, ref_db: Optional[SigapsRefDB] = None) -> None:
        self.ref_db = ref_db or _REF_DB

    def fetch_articles(self, researcher_name: str, limit: int = 50) -> list[Article]:
        try:
            uids = self._search_uids(researcher_name, limit)
            if not uids:
                return []
            return self._fetch_details(uids, researcher_name)
        except requests.RequestException as e:
            logger.warning("PubMed réseau: %s", e)
            return []
        except ET.ParseError as e:
            logger.warning("PubMed XML malformé: %s", e)
            return []

    def _search_uids(self, name: str, limit: int) -> list[str]:
        resp = requests.get(
            f"{self.BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{name}[Author]",
                "retmax": limit,
                "retmode": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _fetch_details(self, uids: list[str], researcher_name: str) -> list[Article]:
        resp = requests.get(
            f"{self.BASE_URL}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(uids), "retmode": "xml"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        target_norm = _normalize_name(researcher_name.split()[-1])
        return [
            art
            for art_tag in root.findall(".//PubmedArticle")
            if (art := self._parse_article(art_tag, target_norm)) is not None
        ]

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
        institution_filter: Optional[str] = None,  # ignoré, conservé pour compat
        limit_per_source: int = 50,
    ) -> list[Article]:
        """Interface compatible avec l'ancien FederatedSearch.search()."""
        articles = self.fetch_articles(name, limit=limit_per_source)
        articles = self._deduplicate(articles)
        articles.sort(key=lambda x: x.publication_year, reverse=True)
        return articles


# Alias rétrocompatible — app.py utilise encore FederatedSearch
FederatedSearch = PubMedFetcher


# ═══════════════════════════════════════════════════════════════════════════════
# RECHERCHE JOURNAL — onglet "🔎 Quel journal choisir ?"
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class JournalResult:
    nlm_id: str
    journal_name: str
    medline_ta: str
    rank: str
    rank_source: str  # "csv" | "heuristique"
    impact_factor: str = ""  # valeur numérique depuis sigaps_ref.csv
    medline_indexed: str = ""  # 'Oui' | 'Non' | '' (depuis sigaps_ref.csv)
    issn: str = ""
    country: str = ""
    similarity: float = 0.0  # score de pertinence agrégé


def fetch_article_by_pmid(
    pmid: str,
    ref_db: Optional[SigapsRefDB] = None,
) -> Optional[dict]:
    """
    Récupère les métadonnées PubMed d'un article par PMID et résout rang + IF.
    """
    db = ref_db or _REF_DB
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": pmid.strip(), "retmode": "xml"},
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

        medline_ta_tag = art_tag.find(".//MedlineJournalInfo/MedlineTA")
        medline_ta = medline_ta_tag.text if medline_ta_tag is not None else "—"

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
    doi: str,
    ref_db: Optional[SigapsRefDB] = None,
) -> Optional[dict]:
    """
    Récupère les métadonnées PubMed d'un article par DOI.
    Stratégie : esearch avec "doi[AID]" → récupère le PMID → efetch XML.
    """
    db = ref_db or _REF_DB
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    doi = (
        doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/").lstrip("doi:")
    )
    try:
        # Étape 1 : DOI → PMID via esearch
        s_resp = requests.get(
            f"{BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{doi}[AID]",
                "retmax": 1,
                "retmode": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        s_resp.raise_for_status()
        ids = s_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            logger.warning("fetch_article_by_doi: aucun PMID pour DOI %s", doi)
            return None
        pmid = ids[0]
        # Étape 2 : PMID → métadonnées via efetch (réutilise la logique existante)
        return fetch_article_by_pmid(pmid, ref_db=db)
    except (requests.RequestException, Exception) as e:
        logger.warning("fetch_article_by_doi(%s): %s", doi, e)
        return None


def _nlm_catalog_search(
    query: str,
    max_results: int = 8,
) -> list[dict]:
    """
    Cherche dans le NLM Catalog (API Entrez) et retourne les résultats bruts.
    Retourne [] en cas d'erreur réseau.
    """
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        s_resp = requests.get(
            f"{BASE}/esearch.fcgi",
            params={
                "db": "nlmcatalog",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        s_resp.raise_for_status()
        uids = s_resp.json().get("esearchresult", {}).get("idlist", [])
        if not uids:
            return []

        f_resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={"db": "nlmcatalog", "id": ",".join(uids), "retmode": "xml"},
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
    query: str,
    ref_db: Optional[SigapsRefDB] = None,
    max_results: int = 8,
) -> list[JournalResult]:
    """
    Double moteur de recherche :

    1. NLM Catalog API (Entrez esearch+efetch) — métadonnées riches (ISSN, pays…)
    2a. Semantic CSV (multilingual-e5-small) — si modèle chargé et embeddings prêts
    2b. Fuzzy CSV (Jaccard trigrammes) — fallback sans modèle

    Fusion par NLM_ID, tri par similarité décroissante.
    """
    db = ref_db or _REF_DB
    model = _EMBED_MODEL

    # ── Encode la requête une seule fois (si modèle disponible) ───────────
    query_vec: Optional[np.ndarray] = None
    if model is not None and db.embeddings_ready:
        query_vec = _e5_encode(model, [query], prefix="query")[0]

    # ── Lancement API en parallèle avec la recherche CSV ──────────────────
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

    # ── Fusion ────────────────────────────────────────────────────────────
    merged: dict[str, JournalResult] = {}

    # -- Résultats API : score de similarité via embedding ou Jaccard --
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

    # -- Résultats CSV (comblent les lacunes de l'API) --
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

    results = sorted(merged.values(), key=lambda x: -x.similarity)
    return results[:max_results]


# ═══════════════════════════════════════════════════════════════════════════════
# SUGGESTION DE JOURNAL PAR TITRE D'ARTICLE
# NLP léger : Jaccard trigrammes titre-par-titre, agrégation pondérée par journal
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SuggestedJournal:
    """
    Journal suggéré à partir de l'analyse des articles similaires sur PubMed.
    similarity = score pondéré normalisé ∈ [0, 1].
    article_count = nombre d'articles similaires publiés dans ce journal.
    """

    nlm_id: str
    journal_name: str
    medline_ta: str
    rank: str
    rank_source: str
    impact_factor: str
    medline_indexed: str
    issn: str
    country: str
    similarity: float  # score agrégé normalisé [0-1]
    article_count: int  # nb articles similaires trouvés dans ce journal
    example_titles: list  # jusqu'à 3 titres d'articles représentatifs
    query_level: int = 1  # 1=titre complet, 2=termes clés, 3=termes réduits
    query_used: str = ""  # requête PubMed effectivement utilisée


# ── Stopwords médicaux pour relaxation de requête ─────────────────────────────

_MEDICAL_STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles, prépositions, conjonctions
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
        # Verbes et mots génériques de titres médicaux
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
        "after",
        "before",
        "among",
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


def _extract_pubmed_query(title: str, min_terms: int = 2) -> list[str]:
    """
    Extrait les termes médicaux significatifs d'un titre d'article.

    Règles (par ordre de priorité) :
    1. Conserver les termes de longueur ≥ 5 et absents des stopwords
    2. Conserver les acronymes médicaux (2-6 majuscules consécutives : AML, HSCT…)
    3. Conserver les mots hybrides avec tiret ou slash (allo-HSCT, AML/MDS)
    4. Éliminer les nombres purs
    5. Si après filtrage il reste < min_terms termes, retourner la liste complète
       pour éviter une requête vide.

    Retourne une liste de tokens prêts à être joints en requête PubMed.
    """
    # Tokenisation sur espaces et ponctuation (sauf tirets et slashes internes)
    tokens = re.findall(r"[\w]+(?:[\-/][\w]+)*", title)

    kept = []
    for tok in tokens:
        # Acronyme médical (ex: AML, HSCT, MDS, BCR-ABL)
        if re.match(r"^[A-Z]{2,6}(?:[\-/][A-Z0-9]{1,6})*$", tok):
            kept.append(tok)
            continue
        # Terme hybride contenant tiret ou slash (allo-HSCT, score-matched)
        if ("-" in tok or "/" in tok) and len(tok) >= 4:
            kept.append(tok)
            continue
        # Terme alphabétique long, hors stopwords
        lower = tok.lower()
        if lower.isalpha() and len(tok) >= 5 and lower not in _MEDICAL_STOPWORDS:
            kept.append(tok)
            continue
        # Nombre pur : ignoré
    # Si trop peu de termes retenus, fallback sur tous les tokens non-stopwords
    if len(kept) < min_terms:
        kept = [
            t for t in tokens if t.lower() not in _MEDICAL_STOPWORDS and not t.isdigit()
        ]
    return kept


# Seuil minimal de résultats PubMed — si en-dessous, on continue la cascade
# même si des résultats ont été trouvés (qualité insuffisante pour e5)
CASCADE_MIN_RESULTS = 15

# Descriptif des 5 niveaux (utilisé dans l'UI pour la bannière de statut)
CASCADE_LEVEL_LABELS = {
    1: ("✅", "#059669", "Titre complet reconnu par PubMed"),
    2: ("🔵", "#2563eb", "Requête relaxée — termes médicaux clés"),
    3: ("⚡", "#d97706", "Tous champs PubMed — 2 termes clés (AND)"),
    4: ("🔁", "#d97706", "Tous champs PubMed — termes élargis (OR)"),
    5: ("⚠️", "#dc2626", "Requête minimale — terme le plus spécifique"),
}


def _build_cascade_queries(title: str) -> list[tuple[str, str]]:
    """
    Cascade de 5 requêtes (term, field_tag) de plus en plus relaxées.

    Niveau 1 : titre complet              + [Title/Abstract]   → match précis
    Niveau 2 : termes médicaux clés (AND) + [Title/Abstract]   → relaxation
    Niveau 3 : 2 termes spécifiques (AND) + tous champs        → moins restrictif
    Niveau 4 : tous termes clés (OR)      + tous champs        → filet large
    Niveau 5 : 1 terme le plus spécifique + tous champs        → filet absolu

    Niveaux 3-5 : aucun field tag → PubMed cherche dans MeSH, abstract,
                  affiliations, keywords, etc.
    Niveau 4 : opérateur OR explicite → union des résultats des termes clés.

    La boucle d'appel continue si le résultat est < CASCADE_MIN_RESULTS,
    ce qui permet de toujours alimenter e5 avec suffisamment d'articles.
    """
    keywords = _extract_pubmed_query(title)
    # Trier par longueur décroissante : les termes longs sont en général plus spécifiques
    kw_sorted = sorted(keywords, key=len, reverse=True)

    q1_term = title.strip()
    q2_term = " ".join(keywords)  # AND implicite
    q3_term = (
        " ".join(kw_sorted[:2])
        if len(kw_sorted) >= 2
        else (kw_sorted[0] if kw_sorted else "")
    )
    q4_term = (
        " OR ".join(kw_sorted[:5]) if kw_sorted else ""
    )  # OR explicite, max 5 termes
    q5_term = kw_sorted[0] if kw_sorted else ""  # terme unique le + spécifique

    raw = [
        (q1_term, "[Title/Abstract]"),
        (q2_term, "[Title/Abstract]"),
        (q3_term, ""),
        (q4_term, ""),
        (q5_term, ""),
    ]

    # Déduplique (même term+field) en conservant l'ordre
    seen: set[tuple[str, str]] = set()
    cascade: list[tuple[str, str]] = []
    for term, field in raw:
        term = term.strip()
        if term and (term, field) not in seen:
            seen.add((term, field))
            cascade.append((term, field))
    return cascade


def suggest_journals_by_title(
    article_title: str,
    ref_db: Optional[SigapsRefDB] = None,
    max_articles: int = 100,
    max_journals: int = 10,
) -> list[SuggestedJournal]:
    """
    Stratégie NLP légère (zéro dépendance externe) :

    1. PubMed esearch sur le titre → max_articles PMIDs
    2. efetch → XML → pour chaque article :
       - titre de l'article PubMed
       - NlmUniqueID du journal
       - Jaccard trigrammes(titre_requête, titre_article) → sim_i
    3. Agrégation par journal :
       weighted_score[j] = Σ sim_i pour tous les articles i publiés dans j
       article_count[j]  = nombre d'articles i publiés dans j
    4. Normalisation : similarity = weighted_score[j] / max(weighted_scores)
    5. Jointure CSV → rang, IF, Medline
    6. Tri par similarity desc, limite max_journals

    Le score reflète à la fois la fréquence ET la pertinence sémantique :
    un journal avec 2 articles très similaires score mieux qu'un journal
    avec 10 articles vaguement liés.
    """
    db = ref_db or _REF_DB
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # ── 1. Recherche PubMed avec cascade de relaxation ────────────────────────
    # Niveau 1 : titre complet  →  Niveau 2 : termes médicaux clés
    # →  Niveau 3 : 3 termes les plus spécifiques
    uids: list[str] = []
    query_used: str = ""
    query_level: int = 0

    for level, (term, field_tag) in enumerate(
        _build_cascade_queries(article_title), start=1
    ):
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
                },
                timeout=REQUEST_TIMEOUT,
            )
            s_resp.raise_for_status()
            candidate_uids = s_resp.json().get("esearchresult", {}).get("idlist", [])
        except requests.RequestException as e:
            logger.warning("suggest_journals esearch level %d: %s", level, e)
            continue

        if candidate_uids:
            # Toujours mémoriser le meilleur résultat obtenu jusqu'ici
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

            # Continuer la cascade si on n'a pas encore assez d'articles pour e5
            if len(uids) >= CASCADE_MIN_RESULTS:
                break
            # Sinon : on continue pour essayer d'obtenir plus de résultats

    if not uids:
        return []

    # ── 2. Récupération des détails XML ──────────────────────────────────────
    try:
        f_resp = requests.get(
            f"{BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(uids), "retmode": "xml"},
            timeout=REQUEST_TIMEOUT,
        )
        f_resp.raise_for_status()
        root = ET.fromstring(f_resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("suggest_journals efetch: %s", e)
        return []

    # ── 3. Collecte des métadonnées des articles ──────────────────────────────
    article_meta: list[dict] = []  # {nlm_id, art_title, jrnl_name, medline_ta, issn}

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
        issn_art_tag = art_tag.find(".//ISSN")
        article_meta.append(
            {
                "nlm_id": (nlm_tag.text or "").strip(),
                "art_title": art_title,
                "jrnl_name": jrnl_tag.text if jrnl_tag is not None else "",
                "medline_ta": mta_tag.text if mta_tag is not None else "—",
                "issn": issn_art_tag.text if issn_art_tag is not None else "",
            }
        )

    if not article_meta:
        return []

    # ── 4. Calcul des similarités ─────────────────────────────────────────────
    model = _EMBED_MODEL
    use_transformer = model is not None and _ST_AVAILABLE

    if use_transformer:
        # Encode en un seul batch : requête en position 0, articles ensuite
        all_texts = [article_title] + [m["art_title"] for m in article_meta]
        all_embs = _e5_encode(
            model,
            all_texts,
            prefix="query",  # même préfixe : on compare des titres à des titres
            normalize=True,
        )
        query_vec_s = all_embs[0]  # shape (384,)
        art_vecs = all_embs[1:]  # shape (N, 384)
        sims = (art_vecs @ query_vec_s).tolist()  # cosine = dot (vecteurs normalisés)
    else:
        # Fallback Jaccard si le modèle n'est pas disponible
        query_tri = _trigrams(article_title)
        sims = [_jaccard(query_tri, _trigrams(m["art_title"])) for m in article_meta]

    # ── 5. Agrégation par journal ─────────────────────────────────────────────
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
        journal_agg[nlm_id]["weighted_score"] += sim
        journal_agg[nlm_id]["count"] += 1
        if len(journal_agg[nlm_id]["example_titles"]) < 3:
            journal_agg[nlm_id]["example_titles"].append(meta["art_title"])

    if not journal_agg:
        return []

    # ── 6. Normalisation ─────────────────────────────────────────────────────
    max_score = max(v["weighted_score"] for v in journal_agg.values()) or 1.0

    # ── 7. Jointure CSV + construction des SuggestedJournal ──────────────────
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

    # ── 8. Tri + limite ──────────────────────────────────────────────────────
    suggestions.sort(key=lambda x: -x.similarity)
    return suggestions[:max_journals]
