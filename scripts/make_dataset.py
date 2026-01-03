import pandas as pd
from pathlib import Path
import sys
import os

# Ajouter le dossier parent au path pour importer 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.cleaning import apply_basic_cleaning

def main():
    # Chemins
    input_path = Path("data/Train.csv")
    output_path = Path("data/train_cleaned.parquet")
    
    # Vérification
    if not input_path.exists():
        print(f"Erreur : Le fichier {input_path} est introuvable.")
        return

    print(f"Chargement de {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Dimensions initiales : {df.shape}")

    # Définition des colonnes
    # Basé sur l'inspection précédente ou la connaissance du dataset
    cat_cols = [
        "Sex", "Marital Status", "Occupation", "P5DA", "RIBP", "NO9I", 
        "FR2B", "VYYP", "1ZJU", "PEK2", "8DMO", "L85J", "A65H", "S9KU", 
        "4K0H", "U37X", "O62J", "OT2U", "V53J", "PI55", "W624", "H53J", "Branch Code"
    ]
    # Note: ci-dessus liste large, cleanons surtout celles qui sont strings
    # On va laisser le script détecter les strings pour normalisation si possible, 
    # mais apply_basic_cleaning demande une liste explicite.
    # Pour faire simple et robuste, on prend les colonnes object.
    
    actual_cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # On retire join_date car traité spécifiquement
    if "join_date" in actual_cat_cols:
        actual_cat_cols.remove("join_date")
    
    print("Nettoyage en cours...")
    df_clean = apply_basic_cleaning(
        df,
        cat_cols=actual_cat_cols
    )
    
    print(f"Sauvegarde vers {output_path}...")
    df_clean.to_parquet(output_path, index=False)
    print("Terminé avec succès.")

if __name__ == "__main__":
    main()
