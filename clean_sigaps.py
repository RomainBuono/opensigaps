import os

import pandas as pd


def clean_sigaps_export(input_file: str, output_file: str):
    print(f"🔄 Traitement de {input_file}...")

    # 1. Détection robuste du début des données
    data_start_row = 0
    encoding = "utf-8"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("⚠️ Encodage UTF-8 échoué, passage en Latin-1...")
        encoding = "latin-1"
        with open(input_file, "r", encoding="latin-1") as f:
            lines = f.readlines()

    found_header = False
    for i, line in enumerate(lines):
        if line.strip().startswith("NLMid"):
            data_start_row = i + 6
            found_header = True
            break

    if not found_header:
        print("❌ Impossible de trouver la ligne 'NLMid'. Vérifiez le fichier.")
        return

    print(f"📍 Données détectées à partir de la ligne {data_start_row}")

    # 2. Chargement "Brut"
    try:
        df = pd.read_csv(
            input_file,
            sep="\t",
            header=None,
            skiprows=data_start_row,
            names=range(20),
            dtype=str,
            encoding=encoding,
            on_bad_lines="skip",
        )
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du CSV : {e}")
        return

    # 3. Extraction et Nettoyage
    df_clean = pd.DataFrame()

    # --- A. Colonne NLM_ID (Index 0) ---
    if 0 in df.columns:
        df_clean["NLM_ID"] = df[0].str.strip()
    else:
        print("❌ Colonne NLM_ID (Index 0) introuvable.")
        return

    # --- B. Colonne Titre (Index 4) ---
    if 4 in df.columns:
        df_clean["Journal"] = df[4].str.strip()
    else:
        print("❌ Colonne Titre (Index 4) introuvable.")
        return

    # --- C. Colonne Indexation Medline (Index 3) ---
    if 3 in df.columns:
        df_clean["Indexation Medline"] = (
            df[3].str.strip().map({"CI": "Oui", "NCI": "Non"}).fillna("Non")
        )
    else:
        df_clean["Indexation Medline"] = "Non"

    # --- D. Colonnes Années (IF et Rank) ---
    years_map = {
        "2020": (8, 9),
        "2021": (10, 11),
        "2022": (12, 13),
        "2023": (14, 15),
        "2024": (16, 17),
    }

    for year, (idx_if, idx_cat) in years_map.items():
        if idx_if in df.columns and idx_cat in df.columns:
            df_clean[f"{year}_IF"] = df[idx_if]
            df_clean[f"{year}_Rank"] = df[idx_cat]
        else:
            df_clean[f"{year}_IF"] = None
            df_clean[f"{year}_Rank"] = "NC"

    # 4. Filtrage et Nettoyage Final
    initial_count = len(df_clean)

    # Suppression des lignes sans NLM_ID (vide ou NaN)
    # On supprime aussi si le NLM_ID est juste un espace vide
    df_clean = df_clean[df_clean["NLM_ID"].str.len() > 0]
    df_clean = df_clean.dropna(subset=["NLM_ID"])

    # Suppression des lignes sans Journal
    df_clean = df_clean.dropna(subset=["Journal"])

    # Remplacement des vides restants par NC dans les colonnes de rang
    df_clean = df_clean.fillna("NC")

    filtered_count = len(df_clean)
    print(
        f"🧹 Filtrage NLM_ID : {initial_count} -> {filtered_count} revues conservées (-{initial_count - filtered_count})."
    )

    # 5. Calcul du "Latest_Rank"
    def get_latest_rank(row):
        for y in ["2024", "2023", "2022", "2021", "2020"]:
            rank_col = f"{y}_Rank"
            if rank_col in row:
                rank = str(row[rank_col]).strip().upper()
                if rank in ["A+", "A", "B", "C", "D", "E"]:
                    return rank
        return "NC"

    df_clean["Latest_Rank"] = df_clean.apply(get_latest_rank, axis=1)

    # 6. Export
    print(f"✅ Export terminé : {len(df_clean)} revues extraites.")
    print("Aperçu des données extraites :")
    print(df_clean[["NLM_ID", "Journal", "Latest_Rank"]].head())

    # Test contrôle
    match = df_clean[
        df_clean["Journal"].str.contains("Blood Adv", case=False, na=False)
    ]
    if not match.empty:
        print("\nTest de contrôle (Blood Adv) :")
        print(match[["NLM_ID", "Journal", "Latest_Rank"]])

    df_clean.to_csv(output_file, index=False)


if __name__ == "__main__":
    if os.path.exists("sigaps.txt"):
        clean_sigaps_export("sigaps.txt", "sigaps_ref.csv")
    else:
        print("❌ Fichier 'sigaps.txt' introuvable.")
