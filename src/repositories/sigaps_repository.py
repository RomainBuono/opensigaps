import csv
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

class SigapsRepository:
    """
    Gère l'accès aux données du référentiel SIGAPS (CSV + Embeddings).
    C'est le seul endroit de l'application qui sait comment lire ces fichiers.
    """
    
    def __init__(self, csv_path: str, npy_path: str, ids_path: str):
        self._csv_path = Path(csv_path)
        self._npy_path = Path(npy_path)
        self._ids_path = Path(ids_path)
        
        self._info_by_key: Dict[str, Dict[str, Any]] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._key_to_idx: Dict[str, int] = {}
        
        self.is_loaded = False

    def load_all(self):
        """Charge le CSV et les matrices numpy en mémoire."""
        self._load_csv()
        self._load_embeddings()
        self.is_loaded = True
        return self

    def _load_csv(self):
        """Lit le CSV et peuple le dictionnaire d'informations."""
        if not self._csv_path.exists():
            print(f"⚠️ Fichier introuvable : {self._csv_path}")
            return
            
        with open(self._csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [str(name).strip() for name in reader.fieldnames] if reader.fieldnames else []
            
            for row in reader:
                nlm_id = row.get("NLM_ID", "").strip()
                if not nlm_id:
                    continue
                    
                self._info_by_key[nlm_id] = {
                    "nlm_id": nlm_id,
                    "journal": row.get("Journal", "").strip(),
                    "rank": row.get("Latest_Rank", "NC").strip(),
                    "impact_factor": row.get("2024_IF", "").strip(),
                    "medline_indexed": row.get("Indexation Medline", "").strip(),
                    "rank_source": "csv"
                }

    def _load_embeddings(self):
        """Lit les fichiers .npy et .ids pré-calculés."""
        if not self._npy_path.exists() or not self._ids_path.exists():
            return
            
        try:
            self._embeddings_matrix = np.load(self._npy_path).astype(np.float32)
            with open(self._ids_path, "r", encoding="utf-8") as f:
                ids = f.read().splitlines()
            self._key_to_idx = {key.strip(): idx for idx, key in enumerate(ids)}
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des embeddings : {e}")

    def get_info(self, key: str) -> Dict[str, Any]:
        """Retourne les métadonnées d'un journal (Rang, IF...)."""
        if not self.is_loaded:
            return {}
        return self._info_by_key.get(key, {})

    def get_journal_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retourne le vecteur pré-calculé d'un journal pour le boost thématique."""
        if not self.is_loaded or self._embeddings_matrix is None:
            return None
        idx = self._key_to_idx.get(key)
        if idx is not None:
            return self._embeddings_matrix[idx]
        return None