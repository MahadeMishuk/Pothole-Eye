import numpy as np
from scipy.optimize import linear_sum_assignment



class KalmanBoxTracker:

    count = 0  #class-level ID counter

    def __init__(self, bbox: np.ndarray):
       
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.no_match_streak = 0    #consecutive frames without detection match
        self.age = 0

        #State dimension: 7 (cx, cy, s, r, dcx, dcy, ds)
        #Measurement: 4 (cx, cy, s, r)
        dim_x, dim_z = 7, 4

        #State transition matrix (constant velocity)
        self.F = np.eye(dim_x)
        self.F[0, 4] = 1  #cx += dcx
        self.F[1, 5] = 1  #cy += dcy
        self.F[2, 6] = 1  #s  += ds

        #Measurement matrix
        self.H = np.eye(dim_z, dim_x)

        #Measurement noise covariance
        self.R = np.eye(dim_z)
        self.R[2, 2] *= 10
        self.R[3, 3] *= 10

        #Process noise covariance
        self.Q = np.eye(dim_x)
        self.Q[4:, 4:] *= 0.01

        #Initial uncertainty
        self.P = np.eye(dim_x)
        self.P[4:, 4:] *= 1000

        #State estimate
        self.x = np.zeros((dim_x, 1))
        self.x[:4] = _box_to_z(bbox)

    def predict(self) -> np.ndarray:
        """Advance state estimate. Returns predicted [x1, y1, x2, y2]."""
        #Prevent negative area
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.no_match_streak > 0:
            self.hits = max(1, self.hits)
        return _z_to_box(self.x[:4])

    def update(self, bbox: np.ndarray):
        """Correct state estimate with new detection."""
        z = _box_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.hits += 1
        self.no_match_streak = 0

    def get_state(self) -> np.ndarray:
        return _z_to_box(self.x[:4])


#Coordinate conversions

def _box_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] → [cx, cy, s, r]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    s = w * h
    r = w / (h + 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)


def _z_to_box(z: np.ndarray) -> np.ndarray:
    """[cx, cy, s, r] → [x1, y1, x2, y2]"""
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    s = max(s, 1.0)
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


#IoU──

def _iou_matrix(detections: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between N detections and M trackers.
    Returns (N, M) matrix.
    """
    n, m = len(detections), len(trackers)
    iou = np.zeros((n, m), dtype=np.float32)
    for i, det in enumerate(detections):
        for j, trk in enumerate(trackers):
            ix1 = max(det[0], trk[0]); iy1 = max(det[1], trk[1])
            ix2 = min(det[2], trk[2]); iy2 = min(det[3], trk[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (det[2] - det[0]) * (det[3] - det[1])
            a2 = (trk[2] - trk[0]) * (trk[3] - trk[1])
            union = a1 + a2 - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def _associate(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float,
):
    """
    Hungarian algorithm matching.
    Returns:
        matches:          list of (det_idx, trk_idx)
        unmatched_dets:   list of det indices
        unmatched_trks:   list of trk indices
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    iou_mat = _iou_matrix(detections, trackers)
    #Hungarian: maximise IoU → minimise negative IoU
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    matches, unmatched_dets, unmatched_trks = [], [], []
    matched_trk = set()
    matched_det = set()

    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] < iou_threshold:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matches.append((r, c))
            matched_det.add(r)
            matched_trk.add(c)

    for i in range(len(detections)):
        if i not in matched_det:
            unmatched_dets.append(i)
    for j in range(len(trackers)):
        if j not in matched_trk:
            unmatched_trks.append(j)

    return matches, unmatched_dets, unmatched_trks


#SORT─

class Sort:
    """
    Multi-object tracker using Kalman filters + Hungarian assignment.

    Usage:
        tracker = Sort(max_age=30, min_hits=2)
        for frame in video:
            dets = np.array([[x1,y1,x2,y2,score], ...])
            tracks = tracker.update(dets)
            #tracks: np.array([[x1,y1,x2,y2,id], ...])
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: np.ndarray shape (N, 5) → [x1, y1, x2, y2, score]
                        Pass np.empty((0, 5)) if no detections.

        Returns:
            np.ndarray shape (M, 5) → [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        #Predict from all active trackers
        predicted = []
        dead = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                dead.append(i)
            else:
                predicted.append(pred)

        #Remove degenerate trackers
        for i in reversed(dead):
            self.trackers.pop(i)

        trk_boxes = np.array(predicted) if predicted else np.empty((0, 4))
        det_boxes = detections[:, :4] if len(detections) else np.empty((0, 4))

        matches, unmatched_dets, unmatched_trks = _associate(
            det_boxes, trk_boxes, self.iou_threshold
        )

        #Update matched trackers
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx, :4])

        #Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[det_idx, :4]))

        #Remove dead trackers (too many missed frames)
        self.trackers = [
            t for t in self.trackers if t.no_match_streak <= self.max_age
        ]
        #Increment no-match streak for unmatched trackers
        matched_trk_indices = {c for _, c in matches}
        for j, trk in enumerate(self.trackers):
            if j not in matched_trk_indices:
                trk.no_match_streak += 1

        #Collect active tracks (min_hits criterion)
        output = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                box = trk.get_state()
                output.append(np.append(box, trk.id))

        return np.array(output) if output else np.empty((0, 5))
