import numpy as np
import json
import os
import traceback

class FishBiomass:
    """
    Expert Module (v2.2) for Sea Bream Biomass Estimation.
    Uses allometric growth formula, pose filtering, and perspective correction.
    """
    def __init__(self, px_to_cm_ratio=0.09, frame_height=1080):
        # Biological Constants for Sparus aurata (Sea Bream)
        self.A = 0.012  # Intercept
        self.B = 3.0    # Growth exponent (Isometric)
        
        # Calibration (Expert v2.2.2: 0.09 is more realistic for this setup)
        self.px_to_cm = px_to_cm_ratio
        self.frame_height = frame_height
        self.y_mid = frame_height / 2
        
        # Perspective Correction Factor (K)
        # EXPERT v2.4: Increased from 0.1 to 0.4 to strongly compensate distance bias
        self.K_DEPTH = 0.4 

        # [EXPERT v2.3.1] Biological Constraints
        self.MIN_FISH_WEIGHT = 100.0  # (g) A sea-bream is rarely below 100g in these cages
        self.MAX_FISH_WEIGHT = 1200.0 # (g) A sea-bream rarely exceeds 1.2kg in aquaculture
        
        # [EXPERT v3.0] ML Mode Setup
        self.MODE_CLASSIC = "CLASSIC"
        self.MODE_ML = "ML-OPTIMIZED"
        self.current_mode = self.MODE_ML
        
        self.ml_coef = None
        self.ml_intercept = None
        # Expert v3.0.4 : Version JSON Ultra-Portable (Zéro Warning)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(os.path.join(current_dir, "..", "weights", "biomass_model.json"))
        self._load_ml_model()

    def _load_ml_model(self):
        """Chargement des coefficients mathématiques (v3.0.4)."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    self.ml_coef = data["coef"]
                    self.ml_intercept = data["intercept"]
                print(f"[BIOMASS] Moteur ML v3.0.4 (JSON) chargé. Précision : {data.get('r2_score', 0):.4f}")
            except Exception as e:
                print(f"[BIOMASS] ERREUR de lecture JSON : {e}")
                traceback.print_exc()
        else:
            print(f"[BIOMASS] Fichier JSON non trouvé à {self.model_path}")

    def toggle_mode(self):
        """Bascule entre le mode classique et le mode ML."""
        if self.current_mode == self.MODE_CLASSIC and self.ml_coef is not None:
            self.current_mode = self.MODE_ML
        else:
            self.current_mode = self.MODE_CLASSIC
        return self.current_mode
        
    def get_pose_score(self, w, h):
        """
        Calculates how 'Sideways' (de profil) a fish is.
        A good profile view usually has Width >> Height for a sea bream.
        """
        if h == 0: return 0
        aspect_ratio = w / h
        # Typical Sea Bream profil is between 1.8 and 2.8 aspect ratio
        if aspect_ratio < 1.4: # Front/Back view (Too narrow)
            return 0.1
        elif 1.8 < aspect_ratio < 2.5:
            return 1.0 # Perfect Profile
        else:
            return 0.5 # Sub-optimal angle
            
    def estimate_weight(self, bbox, frame_h=None):
        """
        Main calculation engine (Expert v2.2).
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        w_px = x2 - x1
        h_px = y2 - y1
        y_center = (y1 + y2) / 2
        
        if frame_h:
            self.frame_height = frame_h
            self.y_mid = frame_h / 2

        # 1. Perspective Correction (Y-Proxy)
        # If fish is at the bottom (y > y_mid), it's close -> reduce its pixels to normalize
        # If fish is at the top (y < y_mid), it's far -> increase its pixels to normalize
        depth_corr = 1.0 - self.K_DEPTH * ((y_center - self.y_mid) / self.frame_height)
        
        # 2. Real-World Dimension Calculation (Projective)
        l_cm = w_px * depth_corr * self.px_to_cm
        h_cm = h_px * depth_corr * self.px_to_cm
        
        # 3. Biomass Estimation (Dual-Mode v3.0)
        if self.current_mode == self.MODE_ML and self.ml_coef is not None:
            # Concept 'Volume Virtuel' : L * H * E (Thickness = 60% Height)
            virtual_volume = l_cm * h_cm * (h_cm * 0.6)
            
            # Calcul manuel Ultra-Pro : y = ax + b
            weight_grams = (self.ml_coef * virtual_volume) + self.ml_intercept
        else:
            # Methode Classic (a * L^b)
            weight_grams = self.A * (l_cm ** self.B)
        
        # 4. Expert v2.3.1 - Hard Clipping (Safety Valve)
        is_clipped = False
        if weight_grams > self.MAX_FISH_WEIGHT:
            weight_grams = self.MAX_FISH_WEIGHT
            is_clipped = True
        elif weight_grams < self.MIN_FISH_WEIGHT:
            weight_grams = self.MIN_FISH_WEIGHT
            is_clipped = True
            
        return weight_grams, self.get_pose_score(w_px, h_px), is_clipped

    def calibrate(self, measured_cm, observed_px):
        """Dynamic calibration based on a reference fish/object."""
        if observed_px == 0: return
        self.px_to_cm = measured_cm / observed_px
        print(f"[BIOMASS] Recalibrated: 1 pixel = {self.px_to_cm:.4f} cm")
