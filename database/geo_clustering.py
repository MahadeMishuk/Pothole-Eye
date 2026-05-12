
import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available — geo clustering will use radius dedup")


@dataclass
class PotholeCluster:
    cluster_id:    str
    center_lat:    float
    center_lon:    float
    severity_max:  float
    severity_avg:  float
    severity_level: str
    count:         int
    confidence:    float
    first_seen:    Optional[float]
    last_seen:     Optional[float]
    member_ids:    List[int]       #DB pothole IDs in this cluster


def cluster_potholes(
    records: list,
    eps_m: float = 3.0,
    min_samples: int = 1,
) -> List[PotholeCluster]:
    """
    Group pothole DB records into spatial clusters.

    Args:
        records:     List of dicts with keys: id, latitude, longitude,
                     severity_score, confidence, first_seen, last_seen
        eps_m:       DBSCAN epsilon radius in metres (default 3m)
        min_samples: Minimum cluster members (1 = no noise rejection)

    Returns:
        List of PotholeCluster objects, sorted by severity_max descending.
    """
    #Filter to records with valid GPS
    geo_records = [r for r in records if r.get("latitude") and r.get("longitude")]
    if not geo_records:
        return []

    if not _SKLEARN_AVAILABLE or len(geo_records) < 2:
        return _single_clusters(geo_records)

    lats = np.array([r["latitude"]  for r in geo_records])
    lons = np.array([r["longitude"] for r in geo_records])

    #Convert lat/lon to local Cartesian metres (accurate for small areas)
    lat_c = float(np.mean(lats))
    lon_c = float(np.mean(lons))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_c))

    coords_m = np.column_stack([
        (lats - lat_c) * m_per_deg_lat,
        (lons - lon_c) * m_per_deg_lon,
    ])

    db = DBSCAN(eps=eps_m, min_samples=min_samples,
                algorithm="ball_tree", metric="euclidean")
    labels = db.fit_predict(coords_m)

    clusters: List[PotholeCluster] = []
    for label in set(labels):
        if label == -1:
            #Noise: treat each as a singleton cluster
            noise_idx = np.where(labels == -1)[0]
            for i in noise_idx:
                r = geo_records[i]
                clusters.append(_single_from_record(r))
            continue

        idx = np.where(labels == label)[0]
        group = [geo_records[i] for i in idx]
        clusters.append(_build_cluster(label, group))

    clusters.sort(key=lambda c: c.severity_max, reverse=True)
    return clusters


def _build_cluster(label: int, group: list) -> PotholeCluster:
    lats  = [r["latitude"]  for r in group]
    lons  = [r["longitude"] for r in group]
    sevs  = [float(r.get("severity_score", 0.0)) for r in group]
    confs = [float(r.get("confidence", 0.5)) for r in group]
    first = min((r["first_seen"] for r in group if r.get("first_seen")), default=None)
    last  = max((r["last_seen"]  for r in group if r.get("last_seen")),  default=None)
    ids   = [r["id"] for r in group if r.get("id")]
    smax  = max(sevs)

    return PotholeCluster(
        cluster_id    = f"cluster_{label}_{int(float(np.mean(lats)) * 1e5)}",
        center_lat    = float(np.mean(lats)),
        center_lon    = float(np.mean(lons)),
        severity_max  = smax,
        severity_avg  = float(np.mean(sevs)),
        severity_level = _level(smax),
        count         = len(group),
        confidence    = float(np.mean(confs)),
        first_seen    = first,
        last_seen     = last,
        member_ids    = ids,
    )


def _single_from_record(r: dict) -> PotholeCluster:
    sev = float(r.get("severity_score", 0.0))
    return PotholeCluster(
        cluster_id    = f"single_{r.get('id', 'x')}",
        center_lat    = float(r["latitude"]),
        center_lon    = float(r["longitude"]),
        severity_max  = sev,
        severity_avg  = sev,
        severity_level = _level(sev),
        count         = 1,
        confidence    = float(r.get("confidence", 0.5)),
        first_seen    = r.get("first_seen"),
        last_seen     = r.get("last_seen"),
        member_ids    = [r["id"]] if r.get("id") else [],
    )


def _single_clusters(records: list) -> List[PotholeCluster]:
    return [_single_from_record(r) for r in records]


def _level(score: float) -> str:
    if score < 0.25: return "L1_COSMETIC"
    if score < 0.50: return "L2_MODERATE"
    if score < 0.75: return "L3_SEVERE"
    return "L4_CRITICAL"
