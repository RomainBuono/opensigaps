import csv
import random
from pathlib import Path

real_csv = Path("data/processed/sigaps_ref.csv")
demo_csv = Path("data/processed/sigaps_demo.csv")

if not real_csv.exists():
    print("Vrai CSV introuvable.")
    exit()

with open(real_csv, 'r', encoding='utf-8-sig') as f:
    reader = list(csv.DictReader(f))
    fieldnames = reader[0].keys()

# On extrait tous les rangs et on les mélange
all_ranks = [row.get("Latest_Rank", "NC") for row in reader]
random.shuffle(all_ranks)

# On réécrit les données avec les rangs truqués
with open(demo_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, row in enumerate(reader):
        row["Latest_Rank"] = all_ranks[i]
        # Optionnel : Mettre des IF aléatoires entre 0 et 20
        if "2022_IF" in row: # Adapte selon le nom de ta colonne
            row["2022_IF"] = round(random.uniform(0.1, 20.0), 3) 
        writer.writerow(row)

print(f"Fichier de démo généré : {demo_csv}")