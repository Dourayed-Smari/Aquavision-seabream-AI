import numpy as np
import os
import time
from .association import *
from .UKF import UKF_Fish

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox, delta_t=1.0, args=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # [x, y, s, score, r, vx, vy, vs, v_score]
        self.kf = UKF_Fish(dim_x=9, dim_z=5, dt=delta_t, args=args)
        
        # Save R_base for NSA Kalman
        self.R_base = self.kf.R.copy()

        # Initial State
        self.kf.x[:5] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_bbox = bbox

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        NSA Kalman logic: adapt R to detection score.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # NSA logic: Trust high-score detections more, increase R for low-score ones
        score = bbox[4]
        adaptive_R = self.R_base * (1.0 - score)
        
        self.kf.update(self.convert_bbox_to_z(bbox), R=adaptive_R)
        self.last_bbox = bbox

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2,score] and returns z in the form
        [center_x, center_y, scale, score, ratio]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    # Scale (Area)
        r = w / float(h + 1e-6) # Ratio
        return np.array([x, y, s, bbox[4], r]).flatten()

    def convert_x_to_bbox(self, x, score=None):
        """
        Takes a state vector x and returns a bounding box in [x1,y1,x2,y2,score]
        """
        w = np.sqrt(x[2] * x[4])
        h = x[2] / (w + 1e-6)
        if score is None: score = x[3]
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

class SUT_Tracker(object):
    def __init__(self, max_age=40, min_hits=3, iou_threshold=0.3, delta_t=0.03):
        """
        Scale-aware Unscented Tracker (SU-T) for Fish.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.trackers = []
        self.frame_count = 0

    def update(self, dets_high=np.empty((0, 5)), dets_low=np.empty((0, 5))):
        """
        Double Cascade Association (ByteTrack Style).
        """
        self.frame_count += 1
        
        # 1. Predict and prepare current trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Management of internal state
        for t in reversed(to_del):
            self.trackers.pop(t)
        if len(to_del) > 0:
            trks = np.delete(trks, to_del, axis=0)

        # 2. First Association : Trackers <-> High Score Detections
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_high, trks, self.iou_threshold)
        
        for m in matched:
            self.trackers[m[1]].update(dets_high[m[0], :])

        # 3. Second Association (The Cascade) : Unmatched Trackers <-> Low Score Detections
        # This helps recover fish that are blurred or partially occluded
        if len(unmatched_trks) > 0 and len(dets_low) > 0:
            remaining_trks = trks[unmatched_trks, :]
            matched_low, unmatched_dets_low, unmatched_trks_final = associate_detections_to_trackers(dets_low, remaining_trks, self.iou_threshold)
            
            for m in matched_low:
                real_trk_idx = unmatched_trks[m[1]]
                self.trackers[real_trk_idx].update(dets_low[m[0], :])
        
        # 4. Initialize New Trackers only from High Score Detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_high[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)
            
        # 5. Output Preparation & Death Management
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # Extrapolation: Show predicted boxes if recently seen (25 frames max for extreme stability)
            if (trk.time_since_update < 25) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1)) 
            i -= 1
            # Remove dead trackers (Increased max_age for long memory)
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # Calculate IoU matrix using FishIoU
    iou_matrix = fish_iou_batch(detections, trackers)
    
    # [NEW] Distance Gating to prevent matching over large distances (fix "ghost lines")
    if iou_matrix.size > 0:
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                # Center distance
                dist = np.linalg.norm((det[:2] + det[2:4]) / 2 - (trk[:2] + trk[2:4]) / 2)
                if dist > 150: # Limit jump to 150 pixels
                    iou_matrix[d, t] = 0.0 # Force low IoU to prevent matching

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
