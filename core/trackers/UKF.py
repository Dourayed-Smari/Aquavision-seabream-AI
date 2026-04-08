import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

class UKF_Fish(UnscentedKalmanFilter):
    """
    Unscented Kalman Filter specifically tuned for Fish Motion.
    9 States: [x, y, s, score, r, vx, vy, vs, v_score]
    5 Observations: [x, y, s, score, r]
    """
    def __init__(self, dim_x=9, dim_z=5, dt=1.0, args=None):
        def fx(x, dt):
            # Constant velocity model for 9 states
            F = np.eye(dim_x)
            for i in range(4): 
                F[i, i+5] = dt
            return np.dot(F, x)

        def hx(x):
            # Observation model: we directly observe the first 5 states
            return x[:5]

        # Use Merwe points for non-linear propagation
        points = MerweScaledSigmaPoints(dim_x, alpha=0.1, beta=2., kappa=0.)
        
        # Fixed: Pass dt to super().__init__
        super().__init__(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=points)
        
        # Custom Uncertainty Tuning for Sea-Bream (v1.6 Ventouse)
        self.P *= 5.0   # Initial uncertainty
        self.R *= 0.5   # Measurement noise (Trust YOLO detections)
        self.Q *= 0.05  # Process noise (Movement agility)
