import os
from roboflow import Roboflow

# =================================================================
# CONFIGURATION CONFIDENTIELLE
# =================================================================
# Remplacez "VOTRE_API_KEY" par votre clé privée Roboflow
# Vous la trouverez ici : https://app.roboflow.com/settings/api
API_KEY = "VOTRE_API_KEY"
PROJECT_NAME = "aquatwins"
DATASET_VERSION = 4 # La version la plus stable pour la segmentation

def download_data():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace("aquatwins").project(PROJECT_NAME)
        
        print(f"--- Téléchargement du dataset {PROJECT_NAME} (v{DATASET_VERSION}) ---")
        dataset = project.version(DATASET_VERSION).download("yolov8")
        
        print(f"\n[SUCCÈS] Dataset téléchargé dans : {dataset.location}")
        print("Note : Vérifiez que les dossiers 'train', 'valid' et 'test' sont présents.")
        
    except Exception as e:
        print(f"\n[ERREUR] Impossible de télécharger les données : {str(e)}")
        print("Avez-vous bien remplacé 'VOTRE_API_KEY' par votre clé réelle ?")

if __name__ == "__main__":
    download_data()
