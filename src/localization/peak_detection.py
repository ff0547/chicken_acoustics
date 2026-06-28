# -*- coding: utf-8 -*-
"""
第2周 Day4：SRP-PHAT 多峰检测模块

功能：
1. 从二维 SRP-PHAT 得分图中提取局部极大值；
2. 使用 NMS 避免同一个声源附近重复出峰；
3. 强制两个预测峰之间保持合理距离，避免两个峰都落在同一条假峰带上；
4. 使用匈牙利算法匹配预测峰和真实源；
5. 统计命中、漏检和虚警。

坐标约定：
- score_map: shape = [ny, nx]
- x_values: shape = [nx]
- y_values: shape = [ny]
- Peak.x / Peak.y 单位：m
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


@dataclass
class Peak:
    """SRP-PHAT 得分图中的一个候选峰。"""

    row: int
    col: int
    x: float
    y: float
    score: float


def normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    """将得分图归一化到 [0, 1]，用于检测和画图。"""
    score_map = np.asarray(score_map, dtype=np.float64)
    finite = np.isfinite(score_map)

    if not np.any(finite):
        return np.zeros_like(score_map, dtype=np.float64)

    valid = score_map[finite]
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    if vmax - vmin < 1e-12:
        return np.zeros_like(score_map, dtype=np.float64)

    out = (score_map - vmin) / (vmax - vmin)
    out[~finite] = 0.0
    return out


def _check_score_map(score_map: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> None:
    """检查得分图和坐标轴是否匹配。"""
    if score_map.ndim != 2:
        raise ValueError("score_map 必须是二维数组。")

    ny, nx = score_map.shape

    if len(x_values) != nx:
        raise ValueError(f"x_values 长度应为 {nx}，当前为 {len(x_values)}。")

    if len(y_values) != ny:
        raise ValueError(f"y_values 长度应为 {ny}，当前为 {len(y_values)}。")


def find_local_peaks(
    score_map: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold_rel: float = 0.06,
    neighborhood_size: int = 3,
    exclude_border: int = 1,
) -> List[Peak]:
    """
    检测局部极大值。

    threshold_rel 不宜太高：
    双源场景中第二个真实峰通常比第一峰弱，太高会直接漏掉第二源。
    """
    score_map = np.asarray(score_map, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)

    _check_score_map(score_map, x_values, y_values)

    if neighborhood_size < 3 or neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size 必须是 >=3 的奇数。")

    finite = np.isfinite(score_map)
    if not np.any(finite):
        return []

    max_score = float(np.nanmax(score_map))
    threshold = max_score * float(threshold_rel)

    pad = neighborhood_size // 2
    padded = np.pad(score_map, pad_width=pad, mode="edge")

    local_max = np.full_like(score_map, -np.inf, dtype=np.float64)

    for dy in range(neighborhood_size):
        for dx in range(neighborhood_size):
            part = padded[dy:dy + score_map.shape[0], dx:dx + score_map.shape[1]]
            local_max = np.maximum(local_max, part)

    peak_mask = (score_map >= local_max - 1e-12) & (score_map >= threshold) & finite

    if exclude_border > 0:
        b = int(exclude_border)
        peak_mask[:b, :] = False
        peak_mask[-b:, :] = False
        peak_mask[:, :b] = False
        peak_mask[:, -b:] = False

    rows_cols = np.argwhere(peak_mask)

    peaks: List[Peak] = []

    for row, col in rows_cols:
        peaks.append(
            Peak(
                row=int(row),
                col=int(col),
                x=float(x_values[col]),
                y=float(y_values[row]),
                score=float(score_map[row, col]),
            )
        )

    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks


def nms_peaks(
    peaks: List[Peak],
    radius_m: float = 0.08,
    max_keep: int = 80,
) -> List[Peak]:
    """
    对候选峰做 NMS。

    Day4 不建议一开始只保留 2 个。
    应该先保留较多候选，再从候选里选两个空间分离的峰。
    """
    if radius_m <= 0:
        raise ValueError("NMS 半径必须大于 0。")

    kept: List[Peak] = []

    for peak in peaks:
        duplicate = False

        for selected in kept:
            dist = float(np.hypot(peak.x - selected.x, peak.y - selected.y))
            if dist < radius_m:
                duplicate = True
                break

        if not duplicate:
            kept.append(peak)

        if len(kept) >= max_keep:
            break

    return kept


def fallback_grid_peaks(
    score_map: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    existing: List[Peak],
    max_keep: int,
    min_distance_m: float,
) -> List[Peak]:
    """
    当局部峰不足时，从全图高分网格点中补充候选峰。
    """
    score_map = np.asarray(score_map, dtype=np.float64)
    flat_order = np.argsort(score_map.ravel())[::-1]

    kept = list(existing)

    for flat_idx in flat_order:
        row, col = np.unravel_index(int(flat_idx), score_map.shape)

        candidate = Peak(
            row=int(row),
            col=int(col),
            x=float(x_values[col]),
            y=float(y_values[row]),
            score=float(score_map[row, col]),
        )

        too_close = False
        for selected in kept:
            dist = float(np.hypot(candidate.x - selected.x, candidate.y - selected.y))
            if dist < min_distance_m:
                too_close = True
                break

        if not too_close:
            kept.append(candidate)

        if len(kept) >= max_keep:
            break

    return kept


def choose_best_separated_pair(
    candidates: List[Peak],
    min_pair_distance_m: float = 0.24,
) -> List[Peak]:
    """
    从候选峰中选择两个空间分离的峰。

    目的：
    你的失败图里 P1、P2 都落在同一条竖向假峰附近。
    因此不能简单取 score 最高的两个局部峰，必须加入预测峰间距约束。

    Day4 真实源间距要求 >= 0.30 m。
    预测峰间距先约束 >= 0.24 m，比较合理。
    """
    if len(candidates) <= 2:
        return candidates

    best_pair = None
    best_value = -np.inf

    top = candidates[:80]

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            p1 = top[i]
            p2 = top[j]

            dist = float(np.hypot(p1.x - p2.x, p1.y - p2.y))

            if dist < min_pair_distance_m:
                continue

            # 分数项：两个峰本身要高。
            score_term = p1.score + p2.score

            # 间距项：避免两个峰挤在一起，但不让距离无限主导。
            sep_bonus = 0.10 * min(dist / 0.50, 1.0)

            value = score_term + sep_bonus

            if value > best_value:
                best_value = value
                best_pair = [p1, p2]

    if best_pair is not None:
        best_pair.sort(key=lambda p: p.score, reverse=True)
        return best_pair

    # 如果严格 0.24 m 找不到，就放宽到 0.18 m。
    relaxed_pair = None
    relaxed_value = -np.inf

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            p1 = top[i]
            p2 = top[j]

            dist = float(np.hypot(p1.x - p2.x, p1.y - p2.y))

            if dist < 0.18:
                continue

            value = p1.score + p2.score

            if value > relaxed_value:
                relaxed_value = value
                relaxed_pair = [p1, p2]

    if relaxed_pair is not None:
        relaxed_pair.sort(key=lambda p: p.score, reverse=True)
        return relaxed_pair

    return candidates[:2]


def detect_top_k_peaks(
    score_map: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    top_k: int = 2,
    threshold_rel: float = 0.06,
    nms_radius_m: float = 0.08,
    neighborhood_size: int = 3,
    exclude_border: int = 1,
    fallback_min_distance_m: float = 0.14,
    min_pair_distance_m: float = 0.24,
) -> List[Peak]:
    """
    从 SRP-PHAT 得分图中提取两个预测峰。

    优化点：
    1. 先保留多个候选峰；
    2. 再选择两个空间分离的峰；
    3. 避免 P1/P2 落在同一条假峰带上。
    """
    score_map = np.asarray(score_map, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)

    _check_score_map(score_map, x_values, y_values)

    peaks = find_local_peaks(
        score_map=score_map,
        x_values=x_values,
        y_values=y_values,
        threshold_rel=threshold_rel,
        neighborhood_size=neighborhood_size,
        exclude_border=exclude_border,
    )

    candidates = nms_peaks(
        peaks=peaks,
        radius_m=nms_radius_m,
        max_keep=80,
    )

    if len(candidates) < top_k:
        candidates = fallback_grid_peaks(
            score_map=score_map,
            x_values=x_values,
            y_values=y_values,
            existing=candidates,
            max_keep=80,
            min_distance_m=fallback_min_distance_m,
        )

    if top_k == 2:
        return choose_best_separated_pair(
            candidates=candidates,
            min_pair_distance_m=min_pair_distance_m,
        )

    return candidates[:top_k]


def peaks_to_xy(peaks: List[Peak]) -> np.ndarray:
    """将 Peak 列表转换为 [N, 2] 坐标数组。"""
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return np.array([[p.x, p.y] for p in peaks], dtype=np.float64)


def _linear_sum_assignment_fallback(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    scipy 不可用时的小规模替代。
    Day4 只有两个真实源和两个预测峰，暴力枚举足够。
    """
    n_rows, n_cols = cost.shape
    k = min(n_rows, n_cols)

    best_rows = None
    best_cols = None
    best_value = np.inf

    for rows in permutations(range(n_rows), k):
        for cols in permutations(range(n_cols), k):
            value = 0.0
            for r, c in zip(rows, cols):
                value += float(cost[r, c])

            if value < best_value:
                best_value = value
                best_rows = rows
                best_cols = cols

    return np.array(best_rows, dtype=int), np.array(best_cols, dtype=int)


def match_sources_hungarian(
    true_xy: np.ndarray,
    pred_xy: np.ndarray,
    hit_radius_m: float = 0.10,
) -> Dict[str, Any]:
    """
    用匈牙利算法匹配真实源和预测峰。

    命中规则：
    - 单源匹配误差 <= hit_radius_m，记为命中；
    - 两个真实源都命中，both_hit=True；
    - 漏检数 = 真实源数 - 命中数；
    - 虚警数 = 预测峰数 - 命中数。
    """
    true_xy = np.asarray(true_xy, dtype=np.float64)
    pred_xy = np.asarray(pred_xy, dtype=np.float64)

    if true_xy.ndim != 2 or true_xy.shape[1] != 2:
        raise ValueError("true_xy 必须为 [N, 2]。")

    if pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
        raise ValueError("pred_xy 必须为 [M, 2]。")

    n_true = int(true_xy.shape[0])
    n_pred = int(pred_xy.shape[0])

    per_true_error_m = np.full(n_true, np.nan, dtype=np.float64)
    per_true_pred_index = np.full(n_true, -1, dtype=np.int64)

    if n_pred == 0:
        return {
            "assignments": [],
            "per_true_error_m": per_true_error_m,
            "per_true_pred_index": per_true_pred_index,
            "num_true": n_true,
            "num_pred": 0,
            "num_hit": 0,
            "miss_count": n_true,
            "false_alarm_count": 0,
            "both_hit": False,
        }

    diff = true_xy[:, None, :] - pred_xy[None, :, :]
    cost = np.sqrt(np.sum(diff ** 2, axis=2))

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        row_ind, col_ind = _linear_sum_assignment_fallback(cost)

    assignments = []
    num_hit = 0

    for r, c in zip(row_ind, col_ind):
        error_m = float(cost[r, c])
        hit = bool(error_m <= hit_radius_m)

        per_true_error_m[int(r)] = error_m
        per_true_pred_index[int(r)] = int(c)

        if hit:
            num_hit += 1

        assignments.append(
            {
                "true_index": int(r),
                "pred_index": int(c),
                "error_m": error_m,
                "hit": hit,
            }
        )

    miss_count = int(n_true - num_hit)
    false_alarm_count = int(n_pred - num_hit)
    both_hit = bool(n_true == 2 and num_hit == 2)

    return {
        "assignments": assignments,
        "per_true_error_m": per_true_error_m,
        "per_true_pred_index": per_true_pred_index,
        "num_true": n_true,
        "num_pred": n_pred,
        "num_hit": int(num_hit),
        "miss_count": miss_count,
        "false_alarm_count": false_alarm_count,
        "both_hit": both_hit,
    }