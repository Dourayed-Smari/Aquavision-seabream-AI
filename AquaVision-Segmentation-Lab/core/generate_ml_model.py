import numpy as np
from sklearn.linear_model import LinearRegression
import json
import os

def generate_reference_model():
    """
    Génère un modèle de régression v3.0.4 - Version Ultra-Portable (JSON).
    Évite les conflits de versions Scikit-Learn en exportant uniquement les coefficients.
    """
    print("[ML-GEN] Génération du modèle Ultra-Portable v3.0.4...")
    
    # 1. Dataset synthétique (Daurade)
    lengths = np.linspace(15, 45, 1000)
    heights = lengths / 2.8 + np.random.normal(0, 0.5, 1000)
    thickness = heights * 0.6
    volumes = lengths * heights * thickness
    weights = 0.012 * (lengths ** 3.0) + np.random.normal(0, 15, 1000)
    
    # 2. Entraînement
    X = volumes.reshape(-1, 1)
    y = weights
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Extraction des coefficients (Pure Math)
    model_data = {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2_score": float(model.score(X, y)),
        "version": "3.0.4-JSON"
    }
    
    # 4. Sauvegarde en JSON
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        
    model_path = os.path.join(weights_dir, "biomass_model.json")
    with open(model_path, "w") as f:
        json.dump(model_data, f, indent=4)
    
    print(f"[ML-GEN] Succès ! Coefficients exportés dans : {model_path}")
    print(f"[ML-GEN] Score R2 : {model_data['r2_score']:.4f}")

if __name__ == "__main__":
    generate_reference_model()
