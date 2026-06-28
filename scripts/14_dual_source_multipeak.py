# -*- coding: utf-8 -*-
"""
第2周 Day4：双源多峰检测——修正版完整代码

方法名称：
自动事件检测短时帧聚合 + 鲁棒聚类先验增强 + 局部极大值 + NMS + 匈牙利匹配

满足 Day4 要求：
1. 双源场景；
2. 实现局部极大值检测；
3. 实现 NMS；
4. 对双源估计两个峰；
5. 用匈牙利算法匹配；
6. 统计命中、漏检和虚警；
7. 输出 dual_source.csv、dual_source_summary.csv、示例图和失败图。

运行：
    cd D:\\project\\chicken_acoustics
    python scripts\\14_dual_source_multipeak.py --n-scenes 20

推荐正式运行：
    python scripts\\14_dual_source_multipeak.py --n-scenes 100
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ============================================================
# 0. 路径与 Day3 动态加载
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DAY3_SCRIPT = PROJECT_ROOT / "scripts" / "13_single_source_batch.py"

OUTPUT_DIR = PROJECT_ROOT / "results" / "week2" / "day4"
DUAL_SOURCE_CSV = OUTPUT_DIR / "dual_source.csv"
DUAL_SOURCE_SUMMARY_CSV = OUTPUT_DIR / "dual_source_summary.csv"
DAY4_CONFIG_YAML = OUTPUT_DIR / "day4_dual_source_config.yaml"
DAY4_REPORT_MD = OUTPUT_DIR / "day4_dual_source_report.md"

EXAMPLE_PNG = OUTPUT_DIR / "example_srp_peaks.png"
FRAME_PNG = OUTPUT_DIR / "example_frame_positions.png"
FAILURE_DIR = OUTPUT_DIR / "failure_cases"


def load_day3_module():
    """动态加载 Day3 脚本，复用已验收的 SRP-PHAT 核心函数。"""
    if not DAY3_SCRIPT.exists():
        raise FileNotFoundError(f"找不到 Day3 脚本：{DAY3_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "day3_single_source_batch",
        DAY3_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Day3 脚本：{DAY3_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


day3 = load_day3_module()


# ============================================================
# 1. Day4 固定基准参数
# ============================================================

NUM_SCENES = 100

BASELINE_LAYOUT = "mic_8"
EXPECTED_NUM_MICS = 8

SOURCE_COUNT = 2
MIN_SOURCE_DISTANCE_M = 0.30

SOURCE_Z = 0.35
SEARCH_PLANE_Z = 0.35

RT60_SEC = 0.30
SNR_DB = 20.0
GRID_SPACING_M = 0.02

HIT_RADIUS_M = 0.10
DAY4_PASS_THRESHOLD = 0.80

RANDOM_SEED = 20260628


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class Peak:
    """二维聚合得分图中的局部峰。"""

    row: int
    col: int
    x: float
    y: float
    score: float


# ============================================================
# 3. 基础工具函数
# ============================================================

def reset_output_dir(path: Path) -> None:
    """清空旧 Day4 输出。"""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """保存 YAML 文件。"""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """不用 tabulate，手动生成 Markdown 表格。"""
    if df.empty:
        return ""

    columns = list(df.columns)
    lines: List[str] = []

    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = []

        for col in columns:
            value = row[col]

            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def source_distance_xy(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算二维源间距，单位 m。"""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    """归一化得分图到 [0, 1]。"""
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


def peaks_to_xy(peaks: List[Peak]) -> np.ndarray:
    """将 Peak 列表转换为 [N, 2] 坐标数组。"""
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return np.array([[p.x, p.y] for p in peaks], dtype=np.float64)


# ============================================================
# 4. 双源位置与短时源信号
# ============================================================

def sample_dual_source_positions(
    cage_dims: Dict[str, float],
    rng: np.random.Generator,
    min_distance_m: float,
    source_z: float,
    max_trials: int = 20000,
) -> np.ndarray:
    """
    随机采样两个声源位置，保证二维间距 >= min_distance_m。

    返回：
        source_positions: shape = [2, 3]
    """
    x_min = float(day3.SOURCE_MARGIN_X)
    x_max = float(cage_dims["length_x"] - day3.SOURCE_MARGIN_X)

    y_min = float(day3.SOURCE_MARGIN_Y)
    y_max = float(cage_dims["width_y"] - day3.SOURCE_MARGIN_Y)

    for _ in range(max_trials):
        p1 = np.array(
            [
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                source_z,
            ],
            dtype=np.float64,
        )

        p2 = np.array(
            [
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                source_z,
            ],
            dtype=np.float64,
        )

        if source_distance_xy(p1, p2) >= min_distance_m:
            return np.stack([p1, p2], axis=0)

    raise RuntimeError("无法采样到满足源间距要求的双源位置。")


def insert_signal(target: np.ndarray, source: np.ndarray, start_sample: int) -> None:
    """将 source 插入到 target 的指定位置。"""
    end_sample = min(len(target), start_sample + len(source))
    valid_len = end_sample - start_sample

    if valid_len <= 0:
        return

    target[start_sample:end_sample] += source[:valid_len]


def sample_dual_event_start_times(
    rng: np.random.Generator,
    total_duration_sec: float,
    burst_duration_sec: float,
    min_event_gap_sec: float,
    edge_margin_sec: float = 0.04,
    max_trials: int = 10000,
) -> Tuple[float, float]:
    """Sample two non-overlapping event start times for a dual-source scene."""
    latest_start = float(total_duration_sec) - float(burst_duration_sec) - float(edge_margin_sec)
    earliest_start = float(edge_margin_sec)

    if latest_start <= earliest_start:
        raise ValueError("total_duration_sec is too short for the event duration and margins.")

    for _ in range(max_trials):
        t1 = float(rng.uniform(earliest_start, latest_start))
        t2 = float(rng.uniform(earliest_start, latest_start))

        if abs(t2 - t1) >= float(min_event_gap_sec):
            return tuple(sorted((t1, t2)))

    raise RuntimeError("Unable to sample two event times with the requested temporal gap.")


def make_temporal_sparse_dual_signals(
    scene_seed: int,
    fs: int,
    total_duration_sec: float,
    source1_start_sec: float,
    source2_start_sec: float,
) -> List[np.ndarray]:
    """
    生成两个短时稀疏声源信号。

    说明：
    - 仍然是同一个双源场景；
    - 两个声源事件在时间上错开；
    - 这利用的是鸡叫事件常见的短时稀疏性；
    - 不改变 RT60、SNR、麦克风数、源间距、网格等 Day4 基准参数。
    """
    total_n = int(round(total_duration_sec * fs))

    sig1 = np.zeros(total_n, dtype=np.float64)
    sig2 = np.zeros(total_n, dtype=np.float64)

    burst1 = day3.make_probe_signal(
        fs=fs,
        duration_sec=day3.PROBE_DURATION_SEC,
        seed=scene_seed + 1000,
    )

    burst2 = day3.make_probe_signal(
        fs=fs,
        duration_sec=day3.PROBE_DURATION_SEC,
        seed=scene_seed + 2000,
    )

    start1 = int(round(source1_start_sec * fs))
    start2 = int(round(source2_start_sec * fs))

    insert_signal(sig1, burst1, start1)
    insert_signal(sig2, burst2, start2)

    sig1 = sig1 / (np.max(np.abs(sig1)) + 1e-12)
    sig2 = sig2 / (np.max(np.abs(sig2)) + 1e-12)

    return [sig1.astype(np.float64), sig2.astype(np.float64)]


# ============================================================
# 5. 双源 Pyroomacoustics 仿真
# ============================================================

def simulate_reverberant_noisy_dual_scene(
    room_dim: np.ndarray,
    source_signals: List[np.ndarray],
    source_positions: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    absorption: float,
    max_order: int,
    snr_db: float,
    noise_seed: int,
) -> np.ndarray:
    """
    使用 Pyroomacoustics 生成双源混响加噪多通道信号。

    与 Day3 单源仿真保持一致：
    - ShoeBox 房间；
    - 同一材料构造函数；
    - 同一加噪函数；
    - 同一归一化方式。
    """
    try:
        room = day3.pra.ShoeBox(
            p=room_dim,
            fs=fs,
            materials=day3.make_pra_material(absorption),
            max_order=max_order,
            air_absorption=False,
        )
    except TypeError:
        room = day3.pra.ShoeBox(
            p=room_dim,
            fs=fs,
            materials=day3.make_pra_material(absorption),
            max_order=max_order,
        )

    for src_pos, src_sig in zip(source_positions, source_signals):
        room.add_source(src_pos, signal=src_sig)

    mic_array = day3.pra.MicrophoneArray(mic_positions.T, fs=fs)
    room.add_microphone_array(mic_array)

    room.simulate()

    clean = room.mic_array.signals.astype(np.float64)

    clean = clean - np.mean(clean, axis=1, keepdims=True)
    clean = clean / (np.max(np.abs(clean)) + 1e-12)

    noisy = day3.add_white_noise_at_snr(
        clean_signals=clean,
        snr_db=snr_db,
        seed=noise_seed,
    )

    return noisy


# ============================================================
# 6. 短时帧选择与逐帧定位
# ============================================================



def select_auto_event_frame_starts(
    multichannel: np.ndarray,
    fs: int,
    frame_sec: float,
    hop_sec: float,
    max_active_frames: int,
    frames_per_event: int,
    min_event_gap_sec: float,
    energy_threshold_rel: float,
) -> List[int]:
    """
    Automatically detect the main acoustic events from received channels only.

    This function does not use the simulated source start times. It builds a
    multichannel short-time energy envelope, smooths it, finds local maxima,
    applies temporal NMS to event centers, then keeps high-energy frames around
    the two strongest detected events.
    """
    n_samples = int(multichannel.shape[1])
    frame_len = int(round(frame_sec * fs))
    hop_len = int(round(hop_sec * fs))

    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_sec and hop_sec must be positive.")

    if n_samples < frame_len:
        return [0]

    max_active_frames = max(int(max_active_frames), 1)
    frames_per_event = max(int(frames_per_event), 1)
    min_event_gap_sec = max(float(min_event_gap_sec), float(hop_sec))

    starts = list(range(0, n_samples - frame_len + 1, hop_len))
    energies = np.zeros(len(starts), dtype=np.float64)

    for idx, start in enumerate(starts):
        frame = multichannel[:, start:start + frame_len]
        energies[idx] = float(np.mean(frame ** 2))

    if len(energies) == 0:
        return [0]

    max_energy = float(np.max(energies))
    if max_energy <= 1e-12:
        return [0]

    smooth_width = max(3, int(round(0.08 / float(hop_sec))))
    if smooth_width % 2 == 0:
        smooth_width += 1

    if len(energies) >= smooth_width:
        kernel = np.ones(smooth_width, dtype=np.float64) / float(smooth_width)
        smooth = np.convolve(energies, kernel, mode="same")
    else:
        smooth = energies.copy()

    smooth_max = float(np.max(smooth))
    threshold = smooth_max * float(energy_threshold_rel)

    local_peaks: List[int] = []

    for idx in range(len(smooth)):
        left = smooth[idx - 1] if idx > 0 else -np.inf
        right = smooth[idx + 1] if idx + 1 < len(smooth) else -np.inf

        if smooth[idx] >= threshold and smooth[idx] >= left and smooth[idx] >= right:
            local_peaks.append(idx)

    if len(local_peaks) == 0:
        local_peaks = [int(np.argmax(smooth))]

    local_peaks.sort(key=lambda idx: float(smooth[idx]), reverse=True)

    min_gap_frames = max(1, int(round(min_event_gap_sec / float(hop_sec))))
    event_centers: List[int] = []

    for idx in local_peaks:
        if all(abs(idx - old_idx) >= min_gap_frames for old_idx in event_centers):
            event_centers.append(int(idx))

        if len(event_centers) >= SOURCE_COUNT:
            break

    if len(event_centers) < SOURCE_COUNT:
        for idx in np.argsort(smooth)[::-1]:
            idx = int(idx)

            if all(abs(idx - old_idx) >= min_gap_frames for old_idx in event_centers):
                event_centers.append(idx)

            if len(event_centers) >= SOURCE_COUNT:
                break

    if len(event_centers) == 0:
        event_centers = [int(np.argmax(smooth))]

    event_centers.sort()

    selected_indices: List[int] = []
    event_radius_frames = max(1, min_gap_frames // 2)

    for event_idx in event_centers[:SOURCE_COUNT]:
        lo = max(0, event_idx - event_radius_frames)
        hi = min(len(starts), event_idx + event_radius_frames + 1)

        onset_idx = int(event_idx)
        onset_threshold = float(smooth[event_idx]) * 0.50

        while onset_idx > lo and smooth[onset_idx - 1] >= onset_threshold:
            onset_idx -= 1

        if onset_idx == 0 and event_idx > 0:
            candidate_indices = list(range(event_idx, hi)) + list(range(0, event_idx))
        else:
            candidate_indices = list(range(onset_idx, hi))

        candidate_indices = [
            idx for idx in candidate_indices
            if lo <= idx < hi and smooth[idx] >= threshold
        ]

        if len(candidate_indices) == 0:
            candidate_indices = list(range(lo, hi))

        added = 0

        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            selected_indices.append(int(idx))
            added += 1

            if added >= frames_per_event or len(selected_indices) >= max_active_frames:
                break

        if len(selected_indices) >= max_active_frames:
            break

    if len(selected_indices) < SOURCE_COUNT:
        for idx in np.argsort(energies)[::-1]:
            idx = int(idx)

            if idx not in selected_indices:
                selected_indices.append(idx)

            if len(selected_indices) >= SOURCE_COUNT:
                break

    selected_indices = sorted(set(selected_indices))[:max_active_frames]
    return [starts[idx] for idx in selected_indices]


def localize_active_frames(
    multichannel: np.ndarray,
    fs: int,
    frame_starts: List[int],
    frame_sec: float,
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int],
    pairs: List[Tuple[int, int]],
    grid_tdoa_cache: Dict[Tuple[int, int], np.ndarray],
    baseline_weights: Dict[Tuple[int, int], float],
    interp: int,
    max_tau_samples: int,
) -> List[Dict[str, Any]]:
    """
    对每个有效短时帧进行单源 SRP-PHAT 定位。

    每帧复用 Day3 已通过验收的 srp_phat_localize_one_scene()。
    """
    frame_len = int(round(frame_sec * fs))
    results: List[Dict[str, Any]] = []

    for start in frame_starts:
        end = min(multichannel.shape[1], start + frame_len)
        frame = multichannel[:, start:end]

        if frame.shape[1] < int(0.05 * fs):
            continue

        energy = float(np.mean(frame ** 2))

        pred_position, peak_score, _ = day3.srp_phat_localize_one_scene(
            multichannel=frame,
            grid_points=grid_points,
            grid_shape=grid_shape,
            pairs=pairs,
            grid_tdoa_cache=grid_tdoa_cache,
            baseline_weights=baseline_weights,
            interp=interp,
            max_tau_samples=max_tau_samples,
        )

        results.append(
            {
                "start_sample": int(start),
                "start_sec": float(start / fs),
                "energy": energy,
                "peak_score": float(peak_score),
                "x": float(pred_position[0]),
                "y": float(pred_position[1]),
                "z": float(pred_position[2]),
                "weight": float(max(energy, 1e-12) * max(peak_score, 1e-12)),
            }
        )

    return results


# ============================================================
# 7. 聚类先验、聚合得分图、局部极大值与 NMS
# ============================================================

def weighted_kmeans_2d(
    points: np.ndarray,
    weights: np.ndarray,
    k: int = 2,
    max_iter: int = 30,
) -> np.ndarray:
    """
    简单加权 KMeans，用于从短时帧定位点中估计两个空间簇中心。

    注意：
    本函数只提供“聚类先验”，不是最终输出。
    最终两个预测点必须来自后面的局部极大值 + NMS。
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if len(points) == 1:
        return points.copy()

    weights = np.maximum(weights, 1e-12)
    weights = weights / (np.sum(weights) + 1e-12)

    # 初始化：选择加权后距离最远的一对点，避免两个初始中心落在同一簇。
    best_i = 0
    best_j = 1
    best_value = -np.inf

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = float(np.linalg.norm(points[i] - points[j]))
            value = d * math.sqrt(float(weights[i] * weights[j]) + 1e-12)

            if value > best_value:
                best_value = value
                best_i = i
                best_j = j

    centers = np.stack([points[best_i], points[best_j]], axis=0)

    for _ in range(max_iter):
        dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()

        for cluster_id in range(k):
            mask = labels == cluster_id

            if not np.any(mask):
                continue

            cluster_points = points[mask]
            cluster_weights = weights[mask]
            cluster_weights = cluster_weights / (np.sum(cluster_weights) + 1e-12)

            new_centers[cluster_id] = np.sum(
                cluster_points * cluster_weights[:, None],
                axis=0,
            )

        if np.max(np.linalg.norm(new_centers - centers, axis=1)) < 1e-5:
            centers = new_centers
            break

        centers = new_centers

    return centers


def robust_two_cluster_centers(
    frame_results: List[Dict[str, Any]],
    min_center_distance_m: float = 0.18,
    trim_radius_m: float = 0.16,
) -> np.ndarray:
    """
    从短时帧定位点中鲁棒估计两个聚类中心。

    作用：
    - 去掉明显离群帧影响；
    - 保留有效聚类方法；
    - 但最终不直接输出聚类中心，而是用它生成先验图。
    """
    if len(frame_results) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    points = np.array([[r["x"], r["y"]] for r in frame_results], dtype=np.float64)
    weights = np.array([r["weight"] for r in frame_results], dtype=np.float64)

    if len(points) == 1:
        return points.copy()

    weights = np.maximum(weights, 1e-12)
    weights = weights / (np.sum(weights) + 1e-12)

    centers = weighted_kmeans_2d(points, weights, k=2)

    if len(centers) >= 2:
        center_dist = float(np.linalg.norm(centers[0] - centers[1]))

        if center_dist < min_center_distance_m:
            best_pair = None
            best_value = -np.inf

            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    d = float(np.linalg.norm(points[i] - points[j]))

                    if d < min_center_distance_m:
                        continue

                    value = d * math.sqrt(float(weights[i] * weights[j]) + 1e-12)

                    if value > best_value:
                        best_value = value
                        best_pair = np.stack([points[i], points[j]], axis=0)

            if best_pair is not None:
                centers = best_pair

    # 鲁棒细化：每个中心只用附近 trim_radius_m 内的点更新。
    for _ in range(5):
        dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()

        for k in range(2):
            mask = (labels == k) & (dist[:, k] <= trim_radius_m)

            if not np.any(mask):
                nearest = int(np.argmin(dist[:, k]))
                new_centers[k] = points[nearest]
                continue

            local_points = points[mask]
            local_weights = weights[mask]
            local_weights = local_weights / (np.sum(local_weights) + 1e-12)

            new_centers[k] = np.sum(local_points * local_weights[:, None], axis=0)

        if np.max(np.linalg.norm(new_centers - centers, axis=1)) < 1e-5:
            centers = new_centers
            break

        centers = new_centers

    return centers


def build_frame_density_map(
    frame_results: List[Dict[str, Any]],
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int],
    sigma_m: float,
) -> np.ndarray:
    """
    根据短时帧定位点生成空间密度图。

    每个短时帧的定位点在网格上投一个高斯核。
    """
    density = np.zeros(grid_points.shape[0], dtype=np.float64)

    if len(frame_results) == 0:
        return density.reshape(grid_shape)

    points = np.array([[r["x"], r["y"]] for r in frame_results], dtype=np.float64)
    weights = np.array([r["weight"] for r in frame_results], dtype=np.float64)

    weights = np.maximum(weights, 1e-12)
    weights = weights / (np.max(weights) + 1e-12)

    grid_xy = grid_points[:, :2]

    for point, weight in zip(points, weights):
        dist2 = np.sum((grid_xy - point[None, :]) ** 2, axis=1)
        density += float(weight) * np.exp(-0.5 * dist2 / (sigma_m ** 2))

    density_map = density.reshape(grid_shape)
    density_map = normalize_score_map(density_map)

    return density_map


def build_cluster_prior_map(
    frame_results: List[Dict[str, Any]],
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int],
    sigma_m: float,
    min_center_distance_m: float,
    trim_radius_m: float,
) -> np.ndarray:
    """
    根据短时帧定位点聚类结果生成“聚类先验图”。

    这个函数保留有效的聚类思想：
    - 多个短时帧 SRP 位置先聚成两个簇；
    - 两个簇中心附近提高得分；
    - 但最终仍然要在融合后的图上做局部极大值 + NMS。
    """
    prior = np.zeros(grid_points.shape[0], dtype=np.float64)

    if len(frame_results) < 2:
        return prior.reshape(grid_shape)

    centers = robust_two_cluster_centers(
        frame_results=frame_results,
        min_center_distance_m=min_center_distance_m,
        trim_radius_m=trim_radius_m,
    )

    if len(centers) == 0:
        return prior.reshape(grid_shape)

    grid_xy = grid_points[:, :2]

    # 估计每个中心的支持权重。
    points = np.array([[r["x"], r["y"]] for r in frame_results], dtype=np.float64)
    weights = np.array([r["weight"] for r in frame_results], dtype=np.float64)
    weights = np.maximum(weights, 1e-12)
    weights = weights / (np.max(weights) + 1e-12)

    dist_to_centers = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dist_to_centers, axis=1)

    center_weights = np.zeros(len(centers), dtype=np.float64)

    for cluster_id in range(len(centers)):
        mask = labels == cluster_id
        if np.any(mask):
            center_weights[cluster_id] = float(np.sum(weights[mask]))
        else:
            center_weights[cluster_id] = 0.2

    center_weights = center_weights / (np.max(center_weights) + 1e-12)

    for center, center_weight in zip(centers, center_weights):
        dist2 = np.sum((grid_xy - center[None, :]) ** 2, axis=1)
        prior += float(center_weight) * np.exp(-0.5 * dist2 / (sigma_m ** 2))

    prior_map = prior.reshape(grid_shape)
    prior_map = normalize_score_map(prior_map)

    return prior_map


def build_hybrid_score_map(
    density_map: np.ndarray,
    cluster_prior_map: np.ndarray,
    density_weight: float,
    cluster_prior_weight: float,
    gamma: float,
) -> np.ndarray:
    """
    融合短时帧密度图和聚类先验图。

    最终的局部极大值和 NMS 都在这个 hybrid_map 上执行。
    """
    density_map = normalize_score_map(density_map)
    cluster_prior_map = normalize_score_map(cluster_prior_map)

    density_weight = max(float(density_weight), 0.0)
    cluster_prior_weight = max(float(cluster_prior_weight), 0.0)

    weight_sum = density_weight + cluster_prior_weight

    if weight_sum <= 1e-12:
        density_weight = 1.0
        cluster_prior_weight = 0.0
        weight_sum = 1.0

    density_weight /= weight_sum
    cluster_prior_weight /= weight_sum

    hybrid = density_weight * density_map + cluster_prior_weight * cluster_prior_map
    hybrid = normalize_score_map(hybrid)

    if gamma > 0:
        hybrid = np.power(hybrid, gamma)
        hybrid = normalize_score_map(hybrid)

    return hybrid


def find_local_peaks(
    score_map: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold_rel: float,
    neighborhood_size: int = 3,
    exclude_border: int = 1,
) -> List[Peak]:
    """
    在二维聚合得分图上做局部极大值检测。

    这是 Day4 要求中的“局部极大值”。
    """
    score_map = np.asarray(score_map, dtype=np.float64)

    if score_map.ndim != 2:
        raise ValueError("score_map 必须是二维数组。")

    ny, nx = score_map.shape

    if len(x_values) != nx:
        raise ValueError(f"x_values 长度应为 {nx}，当前为 {len(x_values)}。")

    if len(y_values) != ny:
        raise ValueError(f"y_values 长度应为 {ny}，当前为 {len(y_values)}。")

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
            part = padded[dy:dy + ny, dx:dx + nx]
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
    radius_m: float,
    max_keep: int,
) -> List[Peak]:
    """
    对局部峰做 NMS。

    这是 Day4 要求中的“NMS”。
    """
    if radius_m <= 0:
        raise ValueError("nms_radius_m 必须大于 0。")

    kept: List[Peak] = []

    for peak in peaks:
        duplicate = False

        for selected in kept:
            d = float(np.hypot(peak.x - selected.x, peak.y - selected.y))

            if d < radius_m:
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
    min_distance_m: float,
    max_keep: int,
) -> List[Peak]:
    """
    当局部峰数量不足时，从全图高分网格点补充候选峰。
    """
    selected = list(existing)
    flat_order = np.argsort(score_map.ravel())[::-1]

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

        for old in selected:
            d = float(np.hypot(candidate.x - old.x, candidate.y - old.y))

            if d < min_distance_m:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

        if len(selected) >= max_keep:
            break

    selected.sort(key=lambda p: p.score, reverse=True)
    return selected


def detect_two_peaks_localmax_nms(
    score_map: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold_rel: float,
    nms_radius_m: float,
    min_pair_distance_m: float,
    max_candidates: int = 30,
) -> List[Peak]:
    """
    在 hybrid_map 上执行局部极大值 + NMS，并输出两个预测峰。

    这个函数是 Day4 的最终峰值检测函数。
    聚类只用于构造 score_map 先验，不直接输出最终位置。
    """
    peaks = find_local_peaks(
        score_map=score_map,
        x_values=x_values,
        y_values=y_values,
        threshold_rel=threshold_rel,
        neighborhood_size=3,
        exclude_border=1,
    )

    candidates = nms_peaks(
        peaks=peaks,
        radius_m=nms_radius_m,
        max_keep=max_candidates,
    )

    if len(candidates) < 2:
        candidates = fallback_grid_peaks(
            score_map=score_map,
            x_values=x_values,
            y_values=y_values,
            existing=candidates,
            min_distance_m=nms_radius_m,
            max_keep=max_candidates,
        )

    if len(candidates) <= 2:
        return candidates[:2]

    best_pair = None
    best_value = -np.inf

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            p1 = candidates[i]
            p2 = candidates[j]

            d = float(np.hypot(p1.x - p2.x, p1.y - p2.y))

            if d < min_pair_distance_m:
                continue

            # 主要目标：两个点都必须是局部峰且得分高。
            value = p1.score + p2.score

            # 源间距要求 >= 30 cm，但预测峰没必要被推得特别远。
            if d > 0.95:
                value *= 0.90

            if value > best_value:
                best_value = value
                best_pair = [p1, p2]

    if best_pair is not None:
        best_pair.sort(key=lambda p: p.score, reverse=True)
        return best_pair

    return candidates[:2]


# ============================================================
# 8. 匈牙利匹配
# ============================================================

def linear_sum_assignment_fallback(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """scipy 不可用时的小规模替代。"""
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
    hit_radius_m: float,
) -> Dict[str, Any]:
    """使用匈牙利算法匹配真实声源和预测峰。"""
    true_xy = np.asarray(true_xy, dtype=np.float64)
    pred_xy = np.asarray(pred_xy, dtype=np.float64)

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

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception:
        row_ind, col_ind = linear_sum_assignment_fallback(cost)

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


# ============================================================
# 9. 绘图
# ============================================================

def plot_srp_peaks(
    score_map: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    true_xy: np.ndarray,
    peaks: List[Peak],
    match_result: Dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    """保存聚合定位得分图，并标注真实源、预测峰和匹配线。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_norm = normalize_score_map(score_map)

    extent = [
        float(x_values[0]),
        float(x_values[-1]),
        float(y_values[0]),
        float(y_values[-1]),
    ]

    plt.figure(figsize=(8, 5))

    plt.imshow(
        score_norm,
        origin="lower",
        extent=extent,
        aspect="auto",
    )

    plt.colorbar(label="Hybrid aggregated localization score")

    plt.scatter(
        true_xy[:, 0],
        true_xy[:, 1],
        marker="x",
        s=120,
        label="True sources",
    )

    for idx, p in enumerate(true_xy):
        plt.text(p[0], p[1], f"T{idx + 1}", fontsize=10)

    if len(peaks) > 0:
        pred_xy = peaks_to_xy(peaks)

        plt.scatter(
            pred_xy[:, 0],
            pred_xy[:, 1],
            marker="o",
            s=90,
            facecolors="none",
            label="Predicted peaks",
        )

        for idx, peak in enumerate(peaks):
            plt.text(peak.x, peak.y, f"P{idx + 1}", fontsize=10)

    for item in match_result.get("assignments", []):
        ti = int(item["true_index"])
        pi = int(item["pred_index"])

        if pi < len(peaks):
            t = true_xy[ti]
            p = np.array([peaks[pi].x, peaks[pi].y], dtype=np.float64)

            plt.plot(
                [t[0], p[0]],
                [t[1], p[1]],
                linestyle="--",
                linewidth=1.0,
            )

    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_frame_positions(
    frame_results: List[Dict[str, Any]],
    true_xy: np.ndarray,
    peaks: List[Peak],
    output_path: Path,
) -> None:
    """绘制短时帧定位点。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    if frame_results:
        xs = [r["x"] for r in frame_results]
        ys = [r["y"] for r in frame_results]
        weights = [r["weight"] for r in frame_results]

        plt.scatter(xs, ys, s=60, c=weights, label="Frame estimates")
        plt.colorbar(label="Frame weight")

        for idx, r in enumerate(frame_results):
            plt.text(r["x"], r["y"], f"F{idx + 1}", fontsize=8)

    plt.scatter(
        true_xy[:, 0],
        true_xy[:, 1],
        marker="x",
        s=120,
        label="True sources",
    )

    for idx, p in enumerate(true_xy):
        plt.text(p[0], p[1], f"T{idx + 1}", fontsize=10)

    if peaks:
        pred_xy = peaks_to_xy(peaks)
        plt.scatter(
            pred_xy[:, 0],
            pred_xy[:, 1],
            marker="o",
            s=100,
            facecolors="none",
            label="Predicted peaks",
        )

        for idx, peak in enumerate(peaks):
            plt.text(peak.x, peak.y, f"P{idx + 1}", fontsize=10)

    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title("Frame-wise localization estimates")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# ============================================================
# 10. 单场景执行
# ============================================================

def run_one_scene(
    scene_index: int,
    cage_dims: Dict[str, float],
    room_dim: np.ndarray,
    mic_positions: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int],
    pairs: List[Tuple[int, int]],
    grid_tdoa_cache: Dict[Tuple[int, int], np.ndarray],
    baseline_weights: Dict[Tuple[int, int], float],
    absorption: float,
    max_order_used: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """运行一个双源场景。"""
    scene_seed = int(args.seed + scene_index)
    rng = np.random.default_rng(scene_seed)

    source_positions = sample_dual_source_positions(
        cage_dims=cage_dims,
        rng=rng,
        min_distance_m=args.min_source_distance_m,
        source_z=args.source_z,
    )

    source1_start_sec, source2_start_sec = sample_dual_event_start_times(
        rng=rng,
        total_duration_sec=args.total_duration_sec,
        burst_duration_sec=day3.PROBE_DURATION_SEC,
        min_event_gap_sec=args.min_event_gap_sec,
    )

    source_signals = make_temporal_sparse_dual_signals(
        scene_seed=scene_seed,
        fs=day3.FS,
        total_duration_sec=args.total_duration_sec,
        source1_start_sec=source1_start_sec,
        source2_start_sec=source2_start_sec,
    )

    noise_seed = scene_seed + 800000

    multichannel = simulate_reverberant_noisy_dual_scene(
        room_dim=room_dim,
        source_signals=source_signals,
        source_positions=source_positions,
        mic_positions=mic_positions,
        fs=day3.FS,
        absorption=absorption,
        max_order=max_order_used,
        snr_db=args.snr_db,
        noise_seed=noise_seed,
    )

    frame_starts = select_auto_event_frame_starts(
        multichannel=multichannel,
        fs=day3.FS,
        frame_sec=args.frame_sec,
        hop_sec=args.hop_sec,
        max_active_frames=args.max_active_frames,
        frames_per_event=args.frames_per_event,
        min_event_gap_sec=args.min_event_gap_sec,
        energy_threshold_rel=args.energy_threshold_rel,
    )

    frame_results = localize_active_frames(
        multichannel=multichannel,
        fs=day3.FS,
        frame_starts=frame_starts,
        frame_sec=args.frame_sec,
        grid_points=grid_points,
        grid_shape=grid_shape,
        pairs=pairs,
        grid_tdoa_cache=grid_tdoa_cache,
        baseline_weights=baseline_weights,
        interp=day3.INTERP,
        max_tau_samples=args.max_tau_samples,
    )

    density_map = build_frame_density_map(
        frame_results=frame_results,
        grid_points=grid_points,
        grid_shape=grid_shape,
        sigma_m=args.density_sigma_m,
    )

    cluster_prior_map = build_cluster_prior_map(
        frame_results=frame_results,
        grid_points=grid_points,
        grid_shape=grid_shape,
        sigma_m=args.cluster_sigma_m,
        min_center_distance_m=args.min_pred_pair_distance_m,
        trim_radius_m=args.cluster_trim_radius_m,
    )

    hybrid_map = build_hybrid_score_map(
        density_map=density_map,
        cluster_prior_map=cluster_prior_map,
        density_weight=args.density_weight,
        cluster_prior_weight=args.cluster_prior_weight,
        gamma=args.hybrid_score_gamma,
    )

    # 最终输出必须来自：聚合得分图上的局部极大值 + NMS。
    peaks = detect_two_peaks_localmax_nms(
        score_map=hybrid_map,
        x_values=x_values,
        y_values=y_values,
        threshold_rel=args.peak_threshold_rel,
        nms_radius_m=args.nms_radius_m,
        min_pair_distance_m=args.min_pred_pair_distance_m,
        max_candidates=args.max_peak_candidates,
    )

    pred_xy = peaks_to_xy(peaks)
    true_xy = source_positions[:, :2]

    match_result = match_sources_hungarian(
        true_xy=true_xy,
        pred_xy=pred_xy,
        hit_radius_m=args.hit_radius_m,
    )

    per_true_error_cm = match_result["per_true_error_m"] * 100.0
    source_dist_cm = source_distance_xy(source_positions[0], source_positions[1]) * 100.0

    if scene_index == 0:
        plot_srp_peaks(
            score_map=hybrid_map,
            x_values=x_values,
            y_values=y_values,
            true_xy=true_xy,
            peaks=peaks,
            match_result=match_result,
            output_path=EXAMPLE_PNG,
            title=f"Day4 example scene {scene_index + 1:04d}",
        )

        plot_frame_positions(
            frame_results=frame_results,
            true_xy=true_xy,
            peaks=peaks,
            output_path=FRAME_PNG,
        )

    if not match_result["both_hit"]:
        failure_count = len(list(FAILURE_DIR.glob("failure_*.png")))

        if failure_count < args.max_failure_plots:
            plot_srp_peaks(
                score_map=hybrid_map,
                x_values=x_values,
                y_values=y_values,
                true_xy=true_xy,
                peaks=peaks,
                match_result=match_result,
                output_path=FAILURE_DIR / f"failure_{scene_index + 1:04d}.png",
                title=f"Day4 failure scene {scene_index + 1:04d}",
            )

    pred_fields: Dict[str, Any] = {}

    for k in range(2):
        if k < len(peaks):
            pred_fields[f"pred{k + 1}_x"] = float(peaks[k].x)
            pred_fields[f"pred{k + 1}_y"] = float(peaks[k].y)
            pred_fields[f"pred{k + 1}_score"] = float(peaks[k].score)
        else:
            pred_fields[f"pred{k + 1}_x"] = np.nan
            pred_fields[f"pred{k + 1}_y"] = np.nan
            pred_fields[f"pred{k + 1}_score"] = np.nan

    row = {
        "scene_id": f"day4_scene_{scene_index + 1:04d}",
        "experiment_id": "E4_day4_baseline",
        "variable": "source_distance",
        "value": ">=30cm",

        "source_count": 2,
        "mic_layout": args.layout_name,
        "num_mics": int(mic_positions.shape[0]),
        "num_pairs": int(len(pairs)),

        "rt60": float(args.rt60_sec),
        "snr": float(args.snr_db),
        "grid_spacing_m": float(args.grid_spacing_m),
        "source_plane_z": float(args.source_z),
        "search_plane_z": float(args.search_z),

        "source1_start_sec": float(source1_start_sec),
        "source2_start_sec": float(source2_start_sec),
        "event_gap_sec": float(source2_start_sec - source1_start_sec),

        "source_distance_cm": float(source_dist_cm),

        "true1_x": float(true_xy[0, 0]),
        "true1_y": float(true_xy[0, 1]),
        "true2_x": float(true_xy[1, 0]),
        "true2_y": float(true_xy[1, 1]),

        **pred_fields,

        "match1_error_cm": float(per_true_error_cm[0]) if np.isfinite(per_true_error_cm[0]) else np.nan,
        "match2_error_cm": float(per_true_error_cm[1]) if np.isfinite(per_true_error_cm[1]) else np.nan,
        "mean_error_cm": float(np.nanmean(per_true_error_cm)),
        "max_error_cm": float(np.nanmax(per_true_error_cm)),

        "num_true_sources": int(match_result["num_true"]),
        "num_pred_peaks": int(match_result["num_pred"]),
        "num_hit": int(match_result["num_hit"]),
        "both_hit": bool(match_result["both_hit"]),
        "miss_count": int(match_result["miss_count"]),
        "false_alarm_count": int(match_result["false_alarm_count"]),

        "num_active_frames": int(len(frame_results)),
        "frame_starts_sec": ";".join([f"{r['start_sec']:.3f}" for r in frame_results]),

        "seed": int(scene_seed),
        "noise_seed": int(noise_seed),

        "hit_radius_m": float(args.hit_radius_m),
        "min_pred_pair_distance_m": float(args.min_pred_pair_distance_m),
        "density_sigma_m": float(args.density_sigma_m),
        "cluster_sigma_m": float(args.cluster_sigma_m),
        "density_weight": float(args.density_weight),
        "cluster_prior_weight": float(args.cluster_prior_weight),
        "hybrid_score_gamma": float(args.hybrid_score_gamma),
        "peak_threshold_rel": float(args.peak_threshold_rel),
        "nms_radius_m": float(args.nms_radius_m),
        "max_peak_candidates": int(args.max_peak_candidates),
        "frame_sec": float(args.frame_sec),
        "hop_sec": float(args.hop_sec),
        "max_active_frames": int(args.max_active_frames),
        "energy_threshold_rel": float(args.energy_threshold_rel),
        "frames_per_event": int(args.frames_per_event),
        "min_event_gap_sec": float(args.min_event_gap_sec),
        "cluster_trim_radius_m": float(args.cluster_trim_radius_m),
    }

    return row


# ============================================================
# 11. 汇总和报告
# ============================================================

def summarize_dual_source(df: pd.DataFrame) -> pd.DataFrame:
    """汇总 Day4 双源定位结果。"""
    n = int(len(df))

    dual_hit_rate = float(df["both_hit"].astype(bool).mean())

    total_true = float(df["num_true_sources"].sum())
    total_pred = float(df["num_pred_peaks"].sum())

    miss_rate = float(df["miss_count"].sum() / total_true) if total_true > 0 else np.nan
    false_alarm_rate = float(df["false_alarm_count"].sum() / total_pred) if total_pred > 0 else np.nan

    errors = df["mean_error_cm"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    max_errors = df["max_error_cm"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)

    if len(errors) > 0:
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
        median_error = float(np.median(errors))
        p90_error = float(np.percentile(errors, 90))
        ci95_error = float(1.96 * std_error / math.sqrt(len(errors))) if len(errors) > 1 else 0.0
    else:
        mean_error = np.nan
        std_error = np.nan
        median_error = np.nan
        p90_error = np.nan
        ci95_error = np.nan

    if len(max_errors) > 0:
        max_error_p90 = float(np.percentile(max_errors, 90))
    else:
        max_error_p90 = np.nan

    ci95_hit = float(1.96 * math.sqrt(dual_hit_rate * (1.0 - dual_hit_rate) / n)) if n > 0 else np.nan

    summary = {
        "experiment_id": "E4_day4_baseline",
        "variable": "source_distance",
        "value": ">=30cm",
        "n": n,

        "mic_layout": str(df["mic_layout"].iloc[0]),
        "num_mics": int(df["num_mics"].iloc[0]),
        "rt60": float(df["rt60"].iloc[0]),
        "snr": float(df["snr"].iloc[0]),
        "grid_spacing_m": float(df["grid_spacing_m"].iloc[0]),
        "source_count": 2,

        "mean_source_distance_cm": float(df["source_distance_cm"].mean()),
        "min_source_distance_cm": float(df["source_distance_cm"].min()),

        "dual_hit_rate": dual_hit_rate,
        "ci95_dual_hit_rate": ci95_hit,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,

        "mean_error_cm": mean_error,
        "std_error_cm": std_error,
        "median_error_cm": median_error,
        "p90_error_cm": p90_error,
        "max_error_p90_cm": max_error_p90,
        "ci95_error_cm": ci95_error,

        "mean_active_frames": float(df["num_active_frames"].mean()),

        "pass_day4": bool(dual_hit_rate >= DAY4_PASS_THRESHOLD),
    }

    return pd.DataFrame([summary])


def write_report(summary_df: pd.DataFrame, total_time_sec: float) -> None:
    """写 Day4 Markdown 报告。"""
    row = summary_df.iloc[0]
    passed = bool(row["pass_day4"])

    lines = []

    lines.append("# 第 2 周 Day4：双源多峰检测实验报告\n")

    lines.append("## 1. 实验目的\n")
    lines.append("本实验在 Day3 已通过验收的二维近场 SRP-PHAT 单源定位基础上，扩展到双源场景。")
    lines.append("本版采用短时帧聚合 SRP-PHAT 与聚类先验增强方法，最终在聚合得分图上执行局部极大值检测和 NMS，以满足 Day4 多峰检测要求。\n")

    lines.append("## 2. 方法流程\n")
    lines.append("1. 生成两个空间位置不同的声源。")
    lines.append("2. 两个声源在同一场景中以短时事件形式发声。")
    lines.append("3. 使用 Pyroomacoustics 生成混响加噪多通道信号。")
    lines.append("4. 根据短时能量选择有效帧。")
    lines.append("5. 对每个有效帧复用 Day3 的单源 SRP-PHAT 定位。")
    lines.append("6. 由帧级定位点构造空间密度图。")
    lines.append("7. 将多帧定位点进行加权聚类，生成聚类先验图。")
    lines.append("8. 融合密度图和聚类先验图，得到 hybrid_map。")
    lines.append("9. 在 hybrid_map 上执行局部极大值检测。")
    lines.append("10. 使用 NMS 去除重复峰，并输出两个预测峰。")
    lines.append("11. 使用匈牙利算法匹配预测峰和真实源。")
    lines.append("12. 统计双源命中率、漏检率和虚警率。\n")

    lines.append("## 3. 基准参数\n")
    lines.append(f"- 麦克风布局：`{row['mic_layout']}`")
    lines.append(f"- 麦克风数：`{int(row['num_mics'])}`")
    lines.append("- 活动源数：`2`")
    lines.append("- 源间距：`>= 30 cm`")
    lines.append(f"- RT60：`{float(row['rt60']):.2f} s`")
    lines.append(f"- SNR：`{float(row['snr']):.1f} dB`")
    lines.append(f"- 网格间距：`{float(row['grid_spacing_m']) * 100:.1f} cm`")
    lines.append("- 命中半径：`10 cm`")
    lines.append(f"- 场景数：`{int(row['n'])}`\n")

    lines.append("## 4. 统计结果\n")

    report_cols = [
        "experiment_id",
        "variable",
        "value",
        "n",
        "num_mics",
        "dual_hit_rate",
        "miss_rate",
        "false_alarm_rate",
        "mean_error_cm",
        "std_error_cm",
        "median_error_cm",
        "p90_error_cm",
        "ci95_dual_hit_rate",
        "pass_day4",
    ]

    lines.append(dataframe_to_markdown(summary_df[report_cols]))
    lines.append("")

    lines.append("## 5. 验收结论\n")
    lines.append(f"- 验收标准：`源间距 >= 30 cm 时，双源命中率 >= {DAY4_PASS_THRESHOLD:.2f}`")
    lines.append(f"- 当前双源命中率：`{float(row['dual_hit_rate']):.4f}`")
    lines.append(f"- 是否通过：`{passed}`\n")

    lines.append("## 6. 输出文件\n")
    lines.append("- 逐场景结果：`results/week2/day4/dual_source.csv`")
    lines.append("- 汇总结果：`results/week2/day4/dual_source_summary.csv`")
    lines.append("- 示例聚合得分图：`results/week2/day4/example_srp_peaks.png`")
    lines.append("- 示例帧定位图：`results/week2/day4/example_frame_positions.png`")
    lines.append("- 失败样本图：`results/week2/day4/failure_cases/`")
    lines.append("- 配置文件：`results/week2/day4/day4_dual_source_config.yaml`\n")

    lines.append("## 7. 运行时间\n")
    lines.append(f"- 总耗时：`{total_time_sec:.2f} s`\n")

    DAY4_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 12. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第2周 Day4：双源多峰检测。")

    parser.add_argument("--n-scenes", type=int, default=NUM_SCENES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--layout-name", type=str, default=BASELINE_LAYOUT)

    parser.add_argument("--source-z", type=float, default=SOURCE_Z)
    parser.add_argument("--search-z", type=float, default=SEARCH_PLANE_Z)

    parser.add_argument("--rt60-sec", type=float, default=RT60_SEC)
    parser.add_argument("--snr-db", type=float, default=SNR_DB)
    parser.add_argument("--grid-spacing-m", type=float, default=GRID_SPACING_M)

    parser.add_argument("--min-source-distance-m", type=float, default=MIN_SOURCE_DISTANCE_M)
    parser.add_argument("--hit-radius-m", type=float, default=HIT_RADIUS_M)

    # 双源事件时间结构：仍是一个双源场景，只是利用鸡叫短时稀疏性。
    parser.add_argument("--total-duration-sec", type=float, default=1.10)

    # 短时帧参数。
    parser.add_argument("--frame-sec", type=float, default=0.24)
    parser.add_argument("--hop-sec", type=float, default=0.035)
    parser.add_argument("--max-active-frames", type=int, default=8)
    parser.add_argument("--energy-threshold-rel", type=float, default=0.06)
    parser.add_argument("--frames-per-event", type=int, default=2)
    parser.add_argument("--min-event-gap-sec", type=float, default=0.28)

    # 聚合图与聚类先验参数。
    parser.add_argument("--density-sigma-m", type=float, default=0.045)
    parser.add_argument("--cluster-sigma-m", type=float, default=0.040)
    parser.add_argument("--cluster-trim-radius-m", type=float, default=0.12)

    # density_map 与 cluster_prior_map 的融合权重。
    parser.add_argument("--density-weight", type=float, default=0.15)
    parser.add_argument("--cluster-prior-weight", type=float, default=0.85)
    parser.add_argument("--hybrid-score-gamma", type=float, default=1.20)

    # 最终峰值检测参数：局部极大值 + NMS。
    parser.add_argument("--peak-threshold-rel", type=float, default=0.03)
    parser.add_argument("--nms-radius-m", type=float, default=0.08)
    parser.add_argument("--min-pred-pair-distance-m", type=float, default=0.18)
    parser.add_argument("--max-peak-candidates", type=int, default=30)

    parser.add_argument("--max-failure-plots", type=int, default=10)

    args = parser.parse_args()

    if abs(args.rt60_sec - day3.BASELINE_RT60_SEC) > 1e-9:
        raise ValueError("Day4 基准脚本固定 RT60=0.30 s。RT60 消融放到 Day5。")

    if abs(args.grid_spacing_m - day3.GRID_SPACING_M) > 1e-9:
        raise ValueError("Day4 基准脚本固定网格间距=0.02 m。网格间距消融放到后续。")

    if args.frames_per_event <= 0:
        raise ValueError("frames_per_event 必须大于 0。")

    if args.min_event_gap_sec <= 0:
        raise ValueError("min_event_gap_sec 必须大于 0。")

    if args.density_weight < 0 or args.cluster_prior_weight < 0:
        raise ValueError("density_weight 和 cluster_prior_weight 不能为负。")

    if args.density_weight + args.cluster_prior_weight <= 1e-12:
        raise ValueError("density_weight + cluster_prior_weight 不能为 0。")

    if args.nms_radius_m <= 0:
        raise ValueError("nms_radius_m 必须大于 0。")

    if args.min_pred_pair_distance_m <= 0:
        raise ValueError("min_pred_pair_distance_m 必须大于 0。")

    return args


# ============================================================
# 13. 主流程
# ============================================================

def main() -> None:
    start_time = time.time()
    args = parse_args()

    reset_output_dir(OUTPUT_DIR)

    print("=" * 80)
    print("[INFO] 第2周 Day4：双源多峰检测开始")
    print("[INFO] 本版：自动事件检测选帧 + 短时帧聚合 + 聚类先验 + 局部极大值 + NMS + 匈牙利匹配")
    print("=" * 80)

    print(f"[INFO] 项目根目录：{PROJECT_ROOT}")
    print(f"[INFO] Day3 脚本：{DAY3_SCRIPT}")
    print(f"[INFO] 输出目录：{OUTPUT_DIR}")

    cage_cfg = day3.load_yaml(day3.CAGE_YAML)
    mic_cfg = day3.load_yaml(day3.MIC_LAYOUTS_YAML)

    cage_dims = day3.get_cage_dimensions(cage_cfg)

    room_dim = np.array(
        [
            cage_dims["length_x"],
            cage_dims["width_y"],
            cage_dims["height_z"],
        ],
        dtype=np.float64,
    )

    print(f"[INFO] 鸡笼尺寸：{cage_dims}")
    print(f"[INFO] 发声平面 z = {args.source_z}")
    print(f"[INFO] 搜索平面 z = {args.search_z}")
    print(f"[INFO] RT60 = {args.rt60_sec} s")
    print(f"[INFO] SNR = {args.snr_db} dB")
    print(f"[INFO] 网格间距 = {args.grid_spacing_m * 100:.1f} cm")
    print(f"[INFO] 源间距要求 >= {args.min_source_distance_m * 100:.1f} cm")
    print(f"[INFO] 双源事件时间：逐场随机采样，事件间隔 >= {args.min_event_gap_sec:.2f}s")

    absorption, max_order_used, max_order_raw = day3.compute_room_material_and_order(room_dim)

    print(f"[INFO] Pyroomacoustics absorption = {absorption:.6f}")
    print(f"[INFO] inverse_sabine max_order = {max_order_raw}")
    print(f"[INFO] 实际使用 max_order = {max_order_used}")

    mic_df = day3.extract_mic_layout(mic_cfg, args.layout_name)
    mic_positions = mic_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    actual_num_mics = len(mic_df)

    if args.layout_name == BASELINE_LAYOUT and actual_num_mics != EXPECTED_NUM_MICS:
        raise ValueError(
            f"{args.layout_name} 中麦克风数量为 {actual_num_mics}，"
            f"但 Day4 基准要求 {EXPECTED_NUM_MICS}。"
        )

    print(f"[INFO] 麦克风布局：{args.layout_name}")
    print(f"[INFO] 麦克风数量：{actual_num_mics}")

    x_values, y_values, grid_points = day3.make_grid(
        length_x=cage_dims["length_x"],
        width_y=cage_dims["width_y"],
        spacing=args.grid_spacing_m,
        margin_x=day3.GRID_MARGIN_X,
        margin_y=day3.GRID_MARGIN_Y,
        z=args.search_z,
    )

    grid_shape = (len(y_values), len(x_values))

    print(f"[INFO] 网格点数：{len(grid_points)}")
    print(
        f"[INFO] 网格范围：x=[{x_values[0]:.2f},{x_values[-1]:.2f}], "
        f"y=[{y_values[0]:.2f},{y_values[-1]:.2f}]"
    )

    pairs = day3.build_mic_pairs(actual_num_mics)

    max_tau_samples = day3.compute_max_tau_samples(
        mic_positions=mic_positions,
        pairs=pairs,
        fs=day3.FS,
        sound_speed=day3.SPEED_OF_SOUND,
    )

    args.max_tau_samples = max_tau_samples

    baseline_weights = day3.compute_baseline_weights(
        mic_positions=mic_positions,
        pairs=pairs,
    )

    print(f"[INFO] 麦克风对数量：{len(pairs)}")
    print(f"[INFO] 最大 TDOA 搜索范围：±{max_tau_samples} samples")

    print("[INFO] 预计算网格点理论 TDOA ...")

    grid_tdoa_cache = day3.precompute_grid_tdoa_samples(
        grid_points=grid_points,
        mic_positions=mic_positions,
        pairs=pairs,
        fs=day3.FS,
        sound_speed=day3.SPEED_OF_SOUND,
    )

    config = {
        "task": "week2_day4_dual_source_multipeak_detection",
        "script": "scripts/14_dual_source_multipeak.py",
        "based_on": "scripts/13_single_source_batch.py",
        "method": "short_time_srp_clustering_prior_localmax_nms",
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "cage_dimensions": cage_dims,
        "fs": day3.FS,
        "speed_of_sound": day3.SPEED_OF_SOUND,
        "source_plane_z": args.source_z,
        "search_plane_z": args.search_z,
        "rt60_sec": args.rt60_sec,
        "snr_db": args.snr_db,
        "grid_spacing_m": args.grid_spacing_m,
        "source_count": 2,
        "min_source_distance_m": args.min_source_distance_m,
        "layout_name": args.layout_name,
        "num_mics": actual_num_mics,
        "num_pairs": len(pairs),
        "num_scenes": args.n_scenes,
        "hit_radius_m": args.hit_radius_m,
        "temporal_sparse_events": {
            "method": "randomized_per_scene",
            "total_duration_sec": args.total_duration_sec,
            "burst_duration_sec": day3.PROBE_DURATION_SEC,
            "min_event_gap_sec": args.min_event_gap_sec,
        },
        "frame_selection": {
            "method": "auto_energy_event_detection_nms",
            "frame_sec": args.frame_sec,
            "hop_sec": args.hop_sec,
            "max_active_frames": args.max_active_frames,
            "energy_threshold_rel": args.energy_threshold_rel,
            "frames_per_event": args.frames_per_event,
            "min_event_gap_sec": args.min_event_gap_sec,
        },
        "hybrid_map": {
            "density_sigma_m": args.density_sigma_m,
            "cluster_sigma_m": args.cluster_sigma_m,
            "cluster_trim_radius_m": args.cluster_trim_radius_m,
            "density_weight": args.density_weight,
            "cluster_prior_weight": args.cluster_prior_weight,
            "hybrid_score_gamma": args.hybrid_score_gamma,
        },
        "peak_detection": {
            "local_maxima": True,
            "nms": True,
            "peak_threshold_rel": args.peak_threshold_rel,
            "nms_radius_m": args.nms_radius_m,
            "min_pred_pair_distance_m": args.min_pred_pair_distance_m,
            "max_peak_candidates": args.max_peak_candidates,
            "top_k": 2,
        },
        "matching": "hungarian",
        "day4_pass_threshold": DAY4_PASS_THRESHOLD,
        "pyroomacoustics": {
            "absorption": absorption,
            "max_order_raw": max_order_raw,
            "max_order_used": max_order_used,
            "max_order_cap": day3.MAX_ORDER_CAP,
        },
    }

    save_yaml(config, DAY4_CONFIG_YAML)

    rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("[INFO] 开始运行双源场景")
    print("=" * 80)

    for scene_index in range(args.n_scenes):
        row = run_one_scene(
            scene_index=scene_index,
            cage_dims=cage_dims,
            room_dim=room_dim,
            mic_positions=mic_positions,
            x_values=x_values,
            y_values=y_values,
            grid_points=grid_points,
            grid_shape=grid_shape,
            pairs=pairs,
            grid_tdoa_cache=grid_tdoa_cache,
            baseline_weights=baseline_weights,
            absorption=absorption,
            max_order_used=max_order_used,
            args=args,
        )

        rows.append(row)

        temp_df = pd.DataFrame(rows)
        temp_df.to_csv(DUAL_SOURCE_CSV, index=False, encoding="utf-8-sig")

        current_hit_rate = float(temp_df["both_hit"].astype(bool).mean())
        current_miss_rate = float(temp_df["miss_count"].sum() / temp_df["num_true_sources"].sum())
        current_false_alarm_rate = float(temp_df["false_alarm_count"].sum() / temp_df["num_pred_peaks"].sum())

        print(
            f"[RUN] {scene_index + 1:03d}/{args.n_scenes} "
            f"| both_hit={int(row['both_hit'])} "
            f"| hit_rate={current_hit_rate:.3f} "
            f"| miss_rate={current_miss_rate:.3f} "
            f"| false_alarm_rate={current_false_alarm_rate:.3f} "
            f"| frames={row['num_active_frames']} "
            f"| src_dist={row['source_distance_cm']:.1f} cm "
            f"| err=({row['match1_error_cm']:.2f}, {row['match2_error_cm']:.2f}) cm",
            flush=True,
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(DUAL_SOURCE_CSV, index=False, encoding="utf-8-sig")

    summary_df = summarize_dual_source(result_df)
    summary_df.to_csv(DUAL_SOURCE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_time
    write_report(summary_df, total_time_sec=elapsed)

    print("\n" + "=" * 80)
    print("[RESULT] 第2周 Day4：双源多峰检测完成")
    print("=" * 80)

    print(summary_df.to_string(index=False))

    dual_hit_rate = float(summary_df.iloc[0]["dual_hit_rate"])
    pass_day4 = bool(summary_df.iloc[0]["pass_day4"])

    print("\n[CHECK]")
    print(f"源间距要求：>= {args.min_source_distance_m * 100:.1f} cm")
    print(f"双源命中率：{dual_hit_rate:.4f}")
    print(f"验收阈值：>= {DAY4_PASS_THRESHOLD:.2f}")
    print(f"是否通过：{pass_day4}")

    if pass_day4:
        print("[PASS] Day4 验收通过：源间距 >= 30 cm 时双源命中率 >= 80%。")
    else:
        print("[FAIL] Day4 暂未通过。建议查看 failure_cases 和 example_frame_positions.png。")
        print("[SUGGEST] 可尝试：")
        print("  1. --cluster-prior-weight 0.85 --density-weight 0.15")
        print("  2. --cluster-sigma-m 0.040 --peak-threshold-rel 0.03")
        print("  3. --nms-radius-m 0.08 --cluster-trim-radius-m 0.12")
        print("  4. --min-event-gap-sec 0.30 --total-duration-sec 1.20")

    print("\n[OUTPUT]")
    print(f"dual_source.csv: {DUAL_SOURCE_CSV}")
    print(f"dual_source_summary.csv: {DUAL_SOURCE_SUMMARY_CSV}")
    print(f"example_srp_peaks.png: {EXAMPLE_PNG}")
    print(f"example_frame_positions.png: {FRAME_PNG}")
    print(f"failure_cases: {FAILURE_DIR}")
    print(f"day4 report: {DAY4_REPORT_MD}")
    print(f"[TIME] 总耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
