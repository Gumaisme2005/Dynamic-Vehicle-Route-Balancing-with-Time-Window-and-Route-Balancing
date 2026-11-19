from bisect import bisect_right
import importlib
from typing import Any, List, Optional, Tuple
import numpy as np


def create_summary_writer(log_dir: str) -> Any:
    """Lazily import SummaryWriter so --no_log works without tensorboard."""
    tb_module = importlib.import_module('torch.utils.tensorboard')
    return tb_module.SummaryWriter(log_dir)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def ndsort_ess(
    objv: np.ndarray,
    need_num: Optional[int] = None,
    need_level: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Efficient Non-dominated Sorting (ENS) for minimization objectives.
    Returns (levels, cri_level) with 1-based front indices and np.inf for unranked items.
    """
    objv = np.asarray(objv, dtype=np.float64)
    n = objv.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64), 0

    m = objv.shape[1]
    order = np.lexsort([objv[:, col] for col in range(m - 1, -1, -1)])

    # Group identical objective vectors so duplicates receive the same front level.
    groups: List[List[int]] = []
    group_vals: List[np.ndarray] = []
    for idx in order:
        val = objv[idx]
        if group_vals and np.array_equal(val, group_vals[-1]):
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
            group_vals.append(val)

    front_assignment = np.empty(len(groups), dtype=np.int64)

    if m == 2:
        fronts_min_f1: List[float] = []
        for g_idx, val in enumerate(group_vals):
            f1 = float(val[1])
            # fronts_min_f1 is monotonically non-decreasing; find first front with min_f1 > f1
            k = bisect_right(fronts_min_f1, f1)
            if k == len(fronts_min_f1):
                fronts_min_f1.append(f1)
            else:
                fronts_min_f1[k] = f1
            front_assignment[g_idx] = k + 1
        max_front = len(fronts_min_f1)
    else:
        fronts_vals: List[List[np.ndarray]] = []
        for g_idx, val in enumerate(group_vals):
            assigned = -1
            for k, front_pts in enumerate(fronts_vals):
                if not any(_dominates(q, val) for q in front_pts):
                    front_pts.append(val)
                    assigned = k
                    break
            if assigned == -1:
                fronts_vals.append([val])
                assigned = len(fronts_vals) - 1
            front_assignment[g_idx] = assigned + 1
        max_front = len(fronts_vals)

    levels = np.full(n, np.inf, dtype=np.float64)
    for g_idx, indices in enumerate(groups):
        levels[indices] = front_assignment[g_idx]

    count = 0
    cri_level = max_front
    for lv in range(1, max_front + 1):
        count += int(np.sum(levels == lv))
        if (need_num is not None and count >= need_num) or (
            need_level is not None and lv >= need_level
        ):
            cri_level = lv
            break

    levels[levels > cri_level] = np.inf
    return levels, int(cri_level)


def crowdis(objv: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """NSGA-II Crowding Distance per front."""
    objv = np.asarray(objv, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    n, m = objv.shape
    dis = np.zeros(n, dtype=np.float64)

    finite_levels = np.unique(levels[np.isfinite(levels)])
    for lv in finite_levels:
        idx = np.flatnonzero(levels == lv)
        if idx.size == 0:
            continue
        if idx.size <= 2:
            dis[idx] = np.inf
            continue

        front_dis = np.zeros(idx.size, dtype=np.float64)
        for obj_i in range(m):
            order = np.argsort(objv[idx, obj_i], kind='mergesort')
            front_dis[order[0]] = np.inf
            front_dis[order[-1]] = np.inf
            f_min = objv[idx[order[0]], obj_i]
            f_max = objv[idx[order[-1]], obj_i]
            if f_max > f_min:
                front_dis[order[1:-1]] += (
                    objv[idx[order[2:]], obj_i] - objv[idx[order[:-2]], obj_i]
                ) / (f_max - f_min)
        dis[idx] = front_dis

    return dis


def _hv_nd_recursive(pts: np.ndarray, ref: np.ndarray) -> float:
    """Exact slicing hypervolume for minimization."""
    if pts.shape[0] == 0:
        return 0.0
    m = pts.shape[1]
    if m == 1:
        return float(ref[0] - np.min(pts[:, 0]))
    if m == 2:
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        pts_sorted = pts[order]
        hv = 0.0
        best_y = float(ref[1])
        ref_x = float(ref[0])
        for x, y in pts_sorted:
            if y < best_y:
                hv += (ref_x - float(x)) * (best_y - float(y))
                best_y = float(y)
        return hv

    order = np.argsort(pts[:, -1])
    pts_sorted = pts[order]
    hv = 0.0
    prev_z = float(ref[-1])
    for i in range(pts_sorted.shape[0] - 1, -1, -1):
        z = float(pts_sorted[i, -1])
        if z < prev_z:
            sub_hv = _hv_nd_recursive(pts_sorted[: i + 1, :-1], ref[:-1])
            hv += sub_hv * (prev_z - z)
            prev_z = z
    return hv


def hv_wfg(objv: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact Hypervolume for minimization given ref_point."""
    objv = np.asarray(objv, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    if objv.size == 0:
        return 0.0

    valid = np.all(objv < ref, axis=1)
    pts = objv[valid]
    if pts.shape[0] == 0:
        return 0.0

    return _hv_nd_recursive(pts, ref)


def hv_indicator(objv: np.ndarray) -> float:
    """Normalized Hypervolume indicator on the non-dominated front when ref_point is None."""
    objv = np.asarray(objv, dtype=np.float64)
    if objv.size == 0:
        return 0.0

    levels, _ = ndsort_ess(objv, need_level=1)
    front = objv[levels == 1]
    if front.shape[0] == 0:
        return 0.0

    min_v = front.min(axis=0)
    max_v = front.max(axis=0)
    span = max_v - min_v
    if np.any(span <= 0):
        return 0.0
    norm_front = (front - min_v) / span
    return hv_wfg(norm_front, np.ones(front.shape[1], dtype=np.float64))
