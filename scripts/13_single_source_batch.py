# -*- coding: utf-8 -*-
"""
第 2 周 Day3：E1 单源批量定位实验——鲁棒优化版

E1 论文级标准：
1. 单源定位；
2. 比较麦克风数量：4 / 6 / 8 / 12；
3. 每组 100 个随机单源场景；
4. 总共 400 次定位；
5. 发声平面：z = 0.35 m；
6. RT60 = 0.30 s；
7. SNR = 20 dB；
8. 网格间距 = 2 cm；
9. 输出 single_source.csv、localization_results.csv、experiment_summary.csv、CDF 图、空间误差图、报告；
10. Day3 验收：基准配置 mic_8 的平均定位误差 <= 10 cm。

本版不修改 E1 要求参数，只优化定位算法：
- GCC-PHAT 正值化与归一化；
- 使用早期分析窗口，降低混响尾部干扰；
- 在候选 TDOA 附近小窗口取最大响应；
- 使用麦克风对可靠性权重；
- 使用麦克风基线长度权重；
- 对 SRP 得分图做轻微空间平滑。

运行：
    cd D:\\project\\chicken_acoustics
    python scripts\\13_single_source_batch.py

输出：
    results/week2/day3/
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

try:
    import pyroomacoustics as pra
except ImportError as exc:
    raise ImportError(
        "当前脚本需要 pyroomacoustics。请先安装：pip install pyroomacoustics"
    ) from exc


# ============================================================
# 1. 全局配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAGE_YAML = PROJECT_ROOT / "configs" / "week1" / "day4" / "cage.yaml"
MIC_LAYOUTS_YAML = PROJECT_ROOT / "configs" / "week1" / "day4" / "mic_layouts.yaml"

OUTPUT_DIR = PROJECT_ROOT / "results" / "week2" / "day3"

# E1：麦克风数量变量
E1_LAYOUTS = [
    ("mic_4", 4),
    ("mic_6", 6),
    ("mic_8", 8),
    ("mic_12", 12),
]

# E1：每组 100 个单源场景
NUM_SCENES_PER_GROUP = 100

# E1：采样率固定 48 kHz
FS = 48_000
SPEED_OF_SOUND = 343.0

# E1：发声平面 z = 0.35 m
SOURCE_Z = 0.35
SEARCH_PLANE_Z = 0.35

# E1：RT60 = 0.30 s，SNR = 20 dB
BASELINE_RT60_SEC = 0.30
BASELINE_SNR_DB = 20.0

# E1：网格间距 2 cm
GRID_SPACING_M = 0.02

# 声源和搜索区域均留边，避免边界假峰。
# 不改变网格间距，只限制到合法发声区域。
SOURCE_MARGIN_X = 0.08
SOURCE_MARGIN_Y = 0.08
GRID_MARGIN_X = SOURCE_MARGIN_X
GRID_MARGIN_Y = SOURCE_MARGIN_Y

# GCC-PHAT 插值倍数
INTERP = 16

# 源信号长度。不是 E1 控制变量，属于仿真探测信号设置。
PROBE_DURATION_SEC = 0.18

# 早期分析窗口。用于降低混响尾部对 GCC-PHAT 的干扰。
ANALYSIS_DURATION_SEC = 0.22

# TDOA 取分窗口：在候选 TDOA 附近 ±2 samples 取最大值。
TDOA_SCORE_WINDOW_SAMPLES = 2.0

# SRP 得分图平滑次数
SMOOTH_PASSES = 1

# Pyroomacoustics 最大镜像源阶数上限。
# RT60 目标值仍为 0.30 s，max_order 只是控制批量仿真复杂度。
MAX_ORDER_CAP = 12

# 固定随机种子
RANDOM_SEED = 20260627

# Day3 验收
BASELINE_LAYOUT = "mic_8"
BASELINE_MEAN_ERROR_THRESHOLD_CM = 10.0

# 输出文件
SINGLE_SOURCE_CSV = OUTPUT_DIR / "single_source.csv"
LOCALIZATION_RESULTS_CSV = OUTPUT_DIR / "localization_results.csv"
EXPERIMENT_SUMMARY_CSV = OUTPUT_DIR / "experiment_summary.csv"
SCENE_POSITIONS_CSV = OUTPUT_DIR / "single_source_positions.csv"

CDF_PNG = OUTPUT_DIR / "single_source_error_cdf.png"
SPATIAL_ERROR_PNG = OUTPUT_DIR / "single_source_spatial_error.png"

REPORT_MD = OUTPUT_DIR / "e1_single_source_report.md"
SCENE_YAML = OUTPUT_DIR / "e1_single_source_scene_config.yaml"


# ============================================================
# 2. 基础工具函数
# ============================================================

def reset_output_dir(path: Path) -> None:
    """删除旧 Day3 结果并重新创建目录。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """保存 YAML 文件。"""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def get_cage_dimensions(cage_cfg: Dict[str, Any]) -> Dict[str, float]:
    """从 cage.yaml 中读取鸡笼尺寸。"""
    dims = cage_cfg.get("dimensions", cage_cfg)

    length_x = float(dims.get("length_x", dims.get("x", dims.get("length", 1.20))))
    width_y = float(dims.get("width_y", dims.get("y", dims.get("width", 0.75))))
    height_z = float(dims.get("height_z", dims.get("z", dims.get("height", 0.60))))

    return {
        "length_x": length_x,
        "width_y": width_y,
        "height_z": height_z,
    }


def extract_mic_layout(mic_cfg: Dict[str, Any], layout_name: str) -> pd.DataFrame:
    """
    从 mic_layouts.yaml 中提取指定麦克风布局。
    """
    layouts = mic_cfg.get("layouts", {})

    if layout_name not in layouts:
        available = list(layouts.keys())
        raise KeyError(
            f"mic_layouts.yaml 中找不到布局：{layout_name}。"
            f"当前可用布局：{available}"
        )

    layout = layouts[layout_name]

    if isinstance(layout, dict):
        mic_obj = (
            layout.get("microphones")
            or layout.get("mics")
            or layout.get("positions")
            or layout.get("mic_positions")
            or layout.get("coordinates")
        )
        if mic_obj is None:
            mic_obj = layout
    else:
        mic_obj = layout

    rows = []

    if isinstance(mic_obj, dict):
        for key, value in mic_obj.items():
            mic_id = str(key)

            if isinstance(value, dict):
                pos = value.get("position") or value.get("pos") or value.get("xyz")
                if pos is None:
                    pos = [value.get("x"), value.get("y"), value.get("z")]
            else:
                pos = value

            if pos is None or len(pos) != 3:
                raise ValueError(f"无法解析麦克风 {mic_id} 的坐标：{value}")

            rows.append({
                "mic_id": mic_id,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            })

    elif isinstance(mic_obj, list):
        for idx, item in enumerate(mic_obj, start=1):
            default_id = f"M{idx:02d}"

            if isinstance(item, dict):
                mic_id = str(
                    item.get("mic_id")
                    or item.get("id")
                    or item.get("name")
                    or default_id
                )

                pos = item.get("position") or item.get("pos") or item.get("xyz")
                if pos is None:
                    pos = [item.get("x"), item.get("y"), item.get("z")]
            else:
                mic_id = default_id
                pos = item

            if pos is None or len(pos) != 3:
                raise ValueError(f"无法解析第 {idx} 个麦克风坐标：{item}")

            rows.append({
                "mic_id": mic_id,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            })

    else:
        raise TypeError(f"无法解析布局 {layout_name} 的数据结构：{type(mic_obj)}")

    mic_df = pd.DataFrame(rows)

    if mic_df.empty:
        raise ValueError(f"布局 {layout_name} 中没有麦克风。")

    mic_df = mic_df.sort_values("mic_id").reset_index(drop=True)
    return mic_df


def make_probe_signal(fs: int, duration_sec: float, seed: int) -> np.ndarray:
    """
    生成确定性宽带源信号。

    使用短时宽带噪声段，便于 GCC-PHAT 形成清晰直达声相关峰。
    这不改变 E1 参数，只是仿真声源信号。
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_sec))

    signal = rng.standard_normal(n)

    fade_len = int(0.015 * fs)
    fade_len = max(1, min(fade_len, n // 8))

    window = np.ones(n)
    fade = np.linspace(0.0, 1.0, fade_len)
    window[:fade_len] = fade
    window[-fade_len:] = fade[::-1]

    signal = signal * window
    signal = signal / (np.max(np.abs(signal)) + 1e-12)

    return signal.astype(np.float64)


def build_mic_pairs(n_mics: int) -> List[Tuple[int, int]]:
    """生成所有麦克风对。"""
    return [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """不用 tabulate，手动生成 Markdown 表格。"""
    if df.empty:
        return ""

    columns = list(df.columns)
    lines = []

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


# ============================================================
# 3. Pyroomacoustics 仿真与加噪
# ============================================================

def compute_room_material_and_order(room_dim: np.ndarray) -> Tuple[float, int, int]:
    """根据目标 RT60 计算吸声系数和最大镜像源阶数。"""
    absorption, max_order_raw = pra.inverse_sabine(BASELINE_RT60_SEC, room_dim)
    max_order_raw = int(max_order_raw)
    max_order_used = min(max_order_raw, MAX_ORDER_CAP)
    return float(absorption), int(max_order_used), int(max_order_raw)


def make_pra_material(absorption: float):
    """创建 Pyroomacoustics 材料对象，兼容不同版本 API。"""
    try:
        return pra.Material(e_absorption=absorption)
    except TypeError:
        pass

    try:
        return pra.Material(energy_absorption=absorption)
    except TypeError:
        pass

    return pra.Material(absorption)


def add_white_noise_at_snr(clean_signals: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """按指定 SNR 加白噪声。"""
    rng = np.random.default_rng(seed)

    signal_power = float(np.mean(clean_signals ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    noise = rng.standard_normal(clean_signals.shape)
    noise = noise / (np.std(noise) + 1e-12)
    noise = noise * math.sqrt(noise_power)

    noisy = clean_signals + noise

    max_abs = np.max(np.abs(noisy))
    if max_abs > 1.0:
        noisy = noisy / (max_abs + 1e-12)

    return noisy.astype(np.float64)


def simulate_reverberant_noisy_scene(
    room_dim: np.ndarray,
    source_signal: np.ndarray,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    absorption: float,
    max_order: int,
    snr_db: float,
    noise_seed: int,
) -> np.ndarray:
    """
    使用 Pyroomacoustics 生成 RT60=0.30 s、SNR=20 dB 的多通道场景。
    """
    try:
        room = pra.ShoeBox(
            p=room_dim,
            fs=fs,
            materials=make_pra_material(absorption),
            max_order=max_order,
            air_absorption=False,
        )
    except TypeError:
        room = pra.ShoeBox(
            p=room_dim,
            fs=fs,
            materials=make_pra_material(absorption),
            max_order=max_order,
        )

    room.add_source(source_position, signal=source_signal)

    mic_array = pra.MicrophoneArray(mic_positions.T, fs=fs)
    room.add_microphone_array(mic_array)

    room.simulate()

    clean = room.mic_array.signals.astype(np.float64)

    clean = clean - np.mean(clean, axis=1, keepdims=True)
    clean = clean / (np.max(np.abs(clean)) + 1e-12)

    noisy = add_white_noise_at_snr(
        clean_signals=clean,
        snr_db=snr_db,
        seed=noise_seed,
    )

    return noisy


# ============================================================
# 4. GCC-PHAT 与 SRP-PHAT 鲁棒定位
# ============================================================

def crop_early_window(multichannel: np.ndarray, fs: int) -> np.ndarray:
    """
    取早期分析窗口。

    原因：
    - RT60=0.30 s 时，后段混响尾部会加强假峰；
    - TDOA 主要由直达声和早期成分决定；
    - 保留早期窗口可以降低混响尾部对 GCC-PHAT 的干扰。
    """
    n = int(round(ANALYSIS_DURATION_SEC * fs))
    n = min(n, multichannel.shape[1])

    x = multichannel[:, :n].copy()
    x = x - np.mean(x, axis=1, keepdims=True)

    peak = np.max(np.abs(x), axis=1, keepdims=True)
    x = x / (peak + 1e-12)

    return x


def gcc_phat_curve(
    sig_i: np.ndarray,
    sig_j: np.ndarray,
    interp: int,
    max_tau_samples: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算鲁棒 GCC-PHAT 曲线。

    优化：
    1. 加 Hann 窗；
    2. PHAT 白化；
    3. 只保留物理 TDOA 范围；
    4. 正值化；
    5. 归一化；
    6. 返回麦克风对可靠性权重。
    """
    sig_i = sig_i - np.mean(sig_i)
    sig_j = sig_j - np.mean(sig_j)

    n0 = min(sig_i.size, sig_j.size)
    sig_i = sig_i[:n0]
    sig_j = sig_j[:n0]

    win = np.hanning(n0)
    sig_i = sig_i * win
    sig_j = sig_j * win

    n_fft = sig_i.size + sig_j.size

    sig_i_fft = np.fft.rfft(sig_i, n=n_fft)
    sig_j_fft = np.fft.rfft(sig_j, n=n_fft)

    cross_power = sig_i_fft * np.conj(sig_j_fft)
    cross_power = cross_power / (np.abs(cross_power) + 1e-12)

    cc_full = np.fft.irfft(cross_power, n=interp * n_fft)

    max_shift = int(interp * n_fft / 2)
    cc_full = np.concatenate((cc_full[-max_shift:], cc_full[:max_shift + 1]))

    lags_samples = np.arange(-max_shift, max_shift + 1, dtype=np.float64) / interp

    keep = np.abs(lags_samples) <= max_tau_samples
    lags_samples = lags_samples[keep]
    cc = cc_full[keep]

    # 关键优化：只保留正相关，避免负响应抵消或误导。
    cc = np.maximum(cc, 0.0)

    # 归一化，使每个麦克风对贡献尺度一致。
    peak = float(np.max(cc))
    if peak > 1e-12:
        cc = cc / peak

    # 可靠性：峰值越突出，背景越低，权重越高。
    p90 = float(np.percentile(cc, 90))
    median = float(np.median(cc))
    reliability = 1.0 / (p90 + 1e-3)

    # 背景太高说明曲线到处都亮，可靠性降低。
    if median > 0.15:
        reliability *= 0.5

    reliability = float(np.clip(reliability, 0.25, 8.0))

    return lags_samples, cc, reliability


def make_grid(
    length_x: float,
    width_y: float,
    spacing: float,
    margin_x: float,
    margin_y: float,
    z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成固定 z 平面的二维搜索网格。"""
    x_values = np.arange(margin_x, length_x - margin_x + 1e-9, spacing)
    y_values = np.arange(margin_y, width_y - margin_y + 1e-9, spacing)

    xx, yy = np.meshgrid(x_values, y_values)

    grid_points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(xx.size, z, dtype=np.float64),
    ])

    return x_values, y_values, grid_points


def compute_max_tau_samples(
    mic_positions: np.ndarray,
    pairs: List[Tuple[int, int]],
    fs: int,
    sound_speed: float,
) -> int:
    """根据阵列最大麦克风间距估计最大 TDOA 范围。"""
    max_mic_dist = 0.0

    for i, j in pairs:
        dist = float(np.linalg.norm(mic_positions[i] - mic_positions[j]))
        max_mic_dist = max(max_mic_dist, dist)

    return int(math.ceil((max_mic_dist / sound_speed) * fs)) + 10


def compute_baseline_weights(
    mic_positions: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """
    计算麦克风对基线权重。

    长基线对 TDOA 的空间分辨力通常更强，因此适当加权。
    """
    baselines = np.array([
        np.linalg.norm(mic_positions[i] - mic_positions[j])
        for i, j in pairs
    ], dtype=np.float64)

    median_baseline = float(np.median(baselines) + 1e-12)

    weights: Dict[Tuple[int, int], float] = {}
    for (i, j), baseline in zip(pairs, baselines):
        w = float(baseline / median_baseline)
        w = float(np.clip(w, 0.5, 2.0))
        weights[(i, j)] = w

    return weights


def precompute_grid_tdoa_samples(
    grid_points: np.ndarray,
    mic_positions: np.ndarray,
    pairs: List[Tuple[int, int]],
    fs: int,
    sound_speed: float,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    预计算每个网格点对每个麦克风对的理论 TDOA。
    """
    cache: Dict[Tuple[int, int], np.ndarray] = {}

    for i, j in pairs:
        mic_i = mic_positions[i]
        mic_j = mic_positions[j]

        di = np.linalg.norm(grid_points - mic_i[None, :], axis=1)
        dj = np.linalg.norm(grid_points - mic_j[None, :], axis=1)

        tdoa_samples = ((di - dj) / sound_speed) * fs
        cache[(i, j)] = tdoa_samples.astype(np.float64)

    return cache


def sample_curve_by_lag_window(
    lags_samples: np.ndarray,
    curve: np.ndarray,
    query_lags_samples: np.ndarray,
    window_samples: float,
) -> np.ndarray:
    """
    在候选 TDOA 附近小窗口内取最大 GCC-PHAT 响应。

    作用：
    - 缓解采样量化误差；
    - 缓解混响导致的轻微峰值偏移；
    - 提高 SRP 对真实位置附近的容忍度。
    """
    lag_min = float(lags_samples[0])
    lag_step = float(lags_samples[1] - lags_samples[0])

    center_index = (query_lags_samples - lag_min) / lag_step
    center_index = np.round(center_index).astype(int)

    half_width = int(round(window_samples / lag_step))
    values = np.zeros_like(query_lags_samples, dtype=np.float64)

    for offset in range(-half_width, half_width + 1):
        idx = center_index + offset
        valid = (idx >= 0) & (idx < len(curve))

        temp = np.zeros_like(query_lags_samples, dtype=np.float64)
        temp[valid] = curve[idx[valid]]

        values = np.maximum(values, temp)

    return values


def smooth_score_map_3x3(score_map: np.ndarray, passes: int) -> np.ndarray:
    """对 SRP 得分图做 3×3 平滑，降低孤立假峰影响。"""
    smoothed = score_map.astype(np.float64)

    for _ in range(passes):
        padded = np.pad(smoothed, pad_width=1, mode="edge")

        smoothed = (
            padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
            padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    return smoothed


def srp_phat_localize_one_scene(
    multichannel: np.ndarray,
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int],
    pairs: List[Tuple[int, int]],
    grid_tdoa_cache: Dict[Tuple[int, int], np.ndarray],
    baseline_weights: Dict[Tuple[int, int], float],
    interp: int,
    max_tau_samples: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    对一个场景执行鲁棒二维 SRP-PHAT 定位。
    """
    x = crop_early_window(multichannel, FS)

    srp_scores = np.zeros(grid_points.shape[0], dtype=np.float64)
    total_weight = 0.0

    for i, j in pairs:
        lags_samples, cc, reliability = gcc_phat_curve(
            sig_i=x[i],
            sig_j=x[j],
            interp=interp,
            max_tau_samples=max_tau_samples,
        )

        query_lags = grid_tdoa_cache[(i, j)]

        pair_score = sample_curve_by_lag_window(
            lags_samples=lags_samples,
            curve=cc,
            query_lags_samples=query_lags,
            window_samples=TDOA_SCORE_WINDOW_SAMPLES,
        )

        weight = reliability * baseline_weights[(i, j)]
        srp_scores += weight * pair_score
        total_weight += weight

    srp_scores = srp_scores / (total_weight + 1e-12)

    score_map = srp_scores.reshape(grid_shape)
    score_map = smooth_score_map_3x3(score_map, passes=SMOOTH_PASSES)

    smoothed_scores = score_map.ravel()

    best_idx = int(np.argmax(smoothed_scores))
    pred_position = grid_points[best_idx]
    peak_score = float(smoothed_scores[best_idx])

    return pred_position, peak_score, smoothed_scores


# ============================================================
# 5. 场景生成、统计与绘图
# ============================================================

def generate_source_positions(
    cage_dims: Dict[str, float],
    num_scenes: int,
    seed: int,
) -> pd.DataFrame:
    """
    生成固定的 100 个单源位置。

    4/6/8/12 麦共用同一批声源位置，保证 E1 对比公平。
    """
    rng = np.random.default_rng(seed)

    x_min = SOURCE_MARGIN_X
    x_max = cage_dims["length_x"] - SOURCE_MARGIN_X
    y_min = SOURCE_MARGIN_Y
    y_max = cage_dims["width_y"] - SOURCE_MARGIN_Y

    xs = rng.uniform(x_min, x_max, size=num_scenes)
    ys = rng.uniform(y_min, y_max, size=num_scenes)
    zs = np.full(num_scenes, SOURCE_Z, dtype=np.float64)

    rows = []

    for idx in range(num_scenes):
        rows.append({
            "scene_id": f"scene_{idx + 1:04d}",
            "source_id": "source_001",
            "true_x": float(xs[idx]),
            "true_y": float(ys[idx]),
            "true_z": float(zs[idx]),
            "seed": int(seed + idx),
        })

    return pd.DataFrame(rows)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    """按麦克风数量汇总 E1 统计指标。"""
    rows = []

    grouped = result_df.groupby(["experiment_id", "variable", "value", "mic_layout", "num_mics"])

    for (experiment_id, variable, value, mic_layout, num_mics), group in grouped:
        errors = group["error_cm"].to_numpy(dtype=np.float64)

        n = len(errors)
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors, ddof=1)) if n > 1 else 0.0
        median_error = float(np.median(errors))
        p90_error = float(np.percentile(errors, 90))
        hit_rate_10cm = float(np.mean(errors <= 10.0))
        hit_rate_20cm = float(np.mean(errors <= 20.0))
        ci95 = float(1.96 * std_error / math.sqrt(n)) if n > 1 else 0.0

        rows.append({
            "experiment_id": experiment_id,
            "variable": variable,
            "value": int(value),
            "mic_layout": mic_layout,
            "num_mics": int(num_mics),
            "n": int(n),

            "mean": mean_error,
            "std": std_error,
            "median": median_error,
            "p90": p90_error,
            "ci95": ci95,

            "mean_error_cm": mean_error,
            "std_error_cm": std_error,
            "median_error_cm": median_error,
            "p90_error_cm": p90_error,
            "hit_rate_10cm": hit_rate_10cm,
            "hit_rate_20cm": hit_rate_20cm,
            "ci95_error_cm": ci95,
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("num_mics").reset_index(drop=True)
    return summary_df


def plot_error_cdf(result_df: pd.DataFrame, output_path: Path) -> None:
    """绘制不同麦克风数量的误差 CDF 曲线。"""
    plt.figure(figsize=(8, 5))

    for num_mics, group in result_df.groupby("num_mics"):
        errors = np.sort(group["error_cm"].to_numpy(dtype=np.float64))
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        plt.plot(errors, cdf, marker=".", linewidth=1.5, label=f"{num_mics} mics")

    plt.axvline(10.0, linestyle="--", linewidth=1.0, label="10 cm")
    plt.axvline(20.0, linestyle=":", linewidth=1.0, label="20 cm")

    plt.xlabel("Localization error / cm")
    plt.ylabel("CDF")
    plt.title("E1 Single-source Localization Error CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_spatial_error(result_df: pd.DataFrame, output_path: Path) -> None:
    """绘制空间误差图。"""
    layouts = sorted(result_df["num_mics"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    vmin = 0.0
    vmax = max(10.0, float(result_df["error_cm"].quantile(0.95)))

    last_scatter = None

    for ax, num_mics in zip(axes, layouts):
        group = result_df[result_df["num_mics"] == num_mics]

        last_scatter = ax.scatter(
            group["true_x"],
            group["true_y"],
            c=group["error_cm"],
            s=28,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"{num_mics} microphones")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.grid(True, alpha=0.3)

    for idx in range(len(layouts), len(axes)):
        axes[idx].axis("off")

    if last_scatter is not None:
        fig.colorbar(last_scatter, ax=axes.tolist(), label="error / cm")

    fig.suptitle("E1 Single-source Spatial Error Map")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    report_path: Path,
    summary_df: pd.DataFrame,
    cage_dims: Dict[str, float],
    total_rows: int,
    baseline_pass: bool,
    absorption: float,
    max_order_used: int,
    max_order_raw: int,
) -> None:
    """写 E1 单源定位实验报告。"""
    baseline_row = summary_df[summary_df["mic_layout"] == BASELINE_LAYOUT]

    if baseline_row.empty:
        baseline_mean = None
        baseline_text = "未找到基准配置 mic_8。"
    else:
        baseline_mean = float(baseline_row.iloc[0]["mean_error_cm"])
        baseline_text = f"{baseline_mean:.3f} cm"

    result_text = "通过" if baseline_pass else "未通过"

    lines = []

    lines.append("# 第 2 周 Day3：E1 单源批量定位实验报告\n")

    lines.append("## 1. 实验目的\n")
    lines.append("本实验对应论文级实验矩阵 E1：单源定位中的麦克风数量影响实验。")
    lines.append("通过比较 4 / 6 / 8 / 12 个麦克风条件下的二维定位误差，评估麦克风数量对 GCC-PHAT + 近场 SRP-PHAT 定位精度的影响。\n")

    lines.append("## 2. 实验设置\n")
    lines.append(f"- 鸡笼局部尺寸：`{cage_dims['length_x']} × {cage_dims['width_y']} × {cage_dims['height_z']} m`")
    lines.append("- 坐标系：`x` 为长度方向，`y` 为深度方向，`z` 为高度方向")
    lines.append(f"- 发声平面：`z = {SOURCE_Z} m`")
    lines.append(f"- 搜索平面：`z = {SEARCH_PLANE_Z} m`")
    lines.append("- 麦克风数量：`4 / 6 / 8 / 12`")
    lines.append(f"- 每组场景数：`{NUM_SCENES_PER_GROUP}`")
    lines.append(f"- 总定位次数：`{total_rows}`")
    lines.append("- 活动源数：`K = 1`")
    lines.append("- 场景类型：`单源、静止声源、几何声学混响、加性白噪声`")
    lines.append(f"- RT60 目标值：`{BASELINE_RT60_SEC} s`")
    lines.append(f"- SNR：`{BASELINE_SNR_DB} dB`")
    lines.append(f"- 采样率：`{FS} Hz`")
    lines.append(f"- 声速：`{SPEED_OF_SOUND} m/s`")
    lines.append(f"- 网格间距：`{GRID_SPACING_M} m`")
    lines.append(f"- GCC-PHAT 插值倍数：`{INTERP}`")
    lines.append(f"- TDOA 取分窗口：`±{TDOA_SCORE_WINDOW_SAMPLES} samples`")
    lines.append(f"- 早期分析窗口：`{ANALYSIS_DURATION_SEC} s`")
    lines.append(f"- Pyroomacoustics 吸声系数：`{absorption:.6f}`")
    lines.append(f"- inverse_sabine 返回 max_order：`{max_order_raw}`")
    lines.append(f"- 实际使用 max_order：`{max_order_used}`\n")

    lines.append("## 3. 鲁棒化处理\n")
    lines.append("- GCC-PHAT 响应正值化与归一化。")
    lines.append("- 仅使用早期分析窗口以降低混响尾部影响。")
    lines.append("- 在候选 TDOA 附近小窗口取最大响应，缓解采样量化误差。")
    lines.append("- 使用麦克风对可靠性权重降低假峰严重麦对的影响。")
    lines.append("- 使用麦克风基线长度权重增强空间分辨力。")
    lines.append("- 对 SRP 得分图做 3×3 平滑，压制孤立假峰。\n")

    lines.append("## 4. E1 统计结果\n")
    report_cols = [
        "experiment_id",
        "variable",
        "value",
        "mic_layout",
        "num_mics",
        "n",
        "mean_error_cm",
        "std_error_cm",
        "median_error_cm",
        "p90_error_cm",
        "hit_rate_10cm",
        "hit_rate_20cm",
        "ci95_error_cm",
    ]
    lines.append(dataframe_to_markdown(summary_df[report_cols]))
    lines.append("")

    lines.append("## 5. Day3 验收结论\n")
    lines.append(f"- 基准配置：`{BASELINE_LAYOUT}`")
    lines.append(f"- 基准配置平均误差：`{baseline_text}`")
    lines.append(f"- 验收标准：`mean_error_cm <= {BASELINE_MEAN_ERROR_THRESHOLD_CM:.1f} cm`")
    lines.append(f"- 验收结论：`{result_text}`\n")

    lines.append("## 6. 仿真边界说明\n")
    lines.append("本阶段仿真采用 Pyroomacoustics 的几何声学 ShoeBox 模型，用于验证多麦克风 TDOA / SRP-PHAT 定位流程。")
    lines.append("模型未精确刻画金属笼条、鸡体遮挡、鸡体散射、衍射、复杂鸡舍设备噪声以及独立 USB 麦克风之间的时钟漂移。")
    lines.append("因此，本实验结果主要反映算法在受控仿真条件下的定位性能，不能直接等同于真实鸡舍部署效果。")
    lines.append("本实验输出的 source_id 为场景内临时声源编号，不对应固定蛋鸡个体，也不建立个体声纹。\n")

    lines.append("## 7. 输出文件\n")
    lines.append("- 单源逐场景结果：`results/week2/day3/single_source.csv`")
    lines.append("- 定位结果表：`results/week2/day3/localization_results.csv`")
    lines.append("- E1 汇总统计表：`results/week2/day3/experiment_summary.csv`")
    lines.append("- 固定声源位置表：`results/week2/day3/single_source_positions.csv`")
    lines.append("- 误差 CDF 图：`results/week2/day3/single_source_error_cdf.png`")
    lines.append("- 空间误差图：`results/week2/day3/single_source_spatial_error.png`")
    lines.append("- 场景配置：`results/week2/day3/e1_single_source_scene_config.yaml`")
    lines.append("- 实验报告：`results/week2/day3/e1_single_source_report.md`\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 6. 主流程
# ============================================================

def main() -> None:
    start_time = time.time()

    reset_output_dir(OUTPUT_DIR)

    print("[INFO] 第 2 周 Day3：E1 单源批量定位实验开始")
    print(f"[INFO] 旧 Day3 结果已清空，输出目录：{OUTPUT_DIR}")

    print("[INFO] 读取 cage.yaml 和 mic_layouts.yaml ...")
    cage_cfg = load_yaml(CAGE_YAML)
    mic_cfg = load_yaml(MIC_LAYOUTS_YAML)

    cage_dims = get_cage_dimensions(cage_cfg)
    room_dim = np.array(
        [cage_dims["length_x"], cage_dims["width_y"], cage_dims["height_z"]],
        dtype=np.float64,
    )

    if not (0.0 < SOURCE_Z < cage_dims["height_z"]):
        raise ValueError(
            f"SOURCE_Z={SOURCE_Z} 超出鸡笼高度范围 0~{cage_dims['height_z']}。"
        )

    absorption, max_order_used, max_order_raw = compute_room_material_and_order(room_dim)

    print(f"[INFO] 鸡笼尺寸：{cage_dims}")
    print(f"[INFO] 发声平面 z = {SOURCE_Z}")
    print(f"[INFO] 搜索平面 z = {SEARCH_PLANE_Z}")
    print(f"[INFO] 目标 RT60 = {BASELINE_RT60_SEC} s")
    print(f"[INFO] SNR = {BASELINE_SNR_DB} dB")
    print(f"[INFO] Pyroomacoustics absorption = {absorption:.6f}")
    print(f"[INFO] inverse_sabine max_order = {max_order_raw}")
    print(f"[INFO] 实际使用 max_order = {max_order_used}")
    print("[INFO] 本版启用：早期窗口 + TDOA窗口取分 + 可靠性加权 + 空间平滑")

    print("[INFO] 生成固定 100 个单源位置，供 4/6/8/12 麦共同使用 ...")
    source_positions_df = generate_source_positions(
        cage_dims=cage_dims,
        num_scenes=NUM_SCENES_PER_GROUP,
        seed=RANDOM_SEED,
    )
    source_positions_df.to_csv(SCENE_POSITIONS_CSV, index=False, encoding="utf-8-sig")

    print("[INFO] 生成二维搜索网格 ...")
    x_values, y_values, grid_points = make_grid(
        length_x=cage_dims["length_x"],
        width_y=cage_dims["width_y"],
        spacing=GRID_SPACING_M,
        margin_x=GRID_MARGIN_X,
        margin_y=GRID_MARGIN_Y,
        z=SEARCH_PLANE_Z,
    )
    grid_shape = (len(y_values), len(x_values))
    print(f"[INFO] 网格点数：{len(grid_points)}")
    print(f"[INFO] 网格范围：x=[{x_values[0]:.2f},{x_values[-1]:.2f}], y=[{y_values[0]:.2f},{y_values[-1]:.2f}]")

    source_signal = make_probe_signal(
        fs=FS,
        duration_sec=PROBE_DURATION_SEC,
        seed=RANDOM_SEED,
    )

    all_rows: List[Dict[str, Any]] = []

    for layout_name, expected_num_mics in E1_LAYOUTS:
        print("\n" + "=" * 80)
        print(f"[INFO] 开始 E1 组：{layout_name}，期望麦克风数量：{expected_num_mics}")

        mic_df = extract_mic_layout(mic_cfg, layout_name)
        mic_positions = mic_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

        actual_num_mics = len(mic_df)

        if actual_num_mics != expected_num_mics:
            raise ValueError(
                f"{layout_name} 中麦克风数量为 {actual_num_mics}，"
                f"但 E1 期望为 {expected_num_mics}。请检查 mic_layouts.yaml。"
            )

        if np.any(mic_positions[:, 2] >= cage_dims["height_z"]):
            raise ValueError(
                f"{layout_name} 中存在麦克风 z 坐标超出鸡笼高度，请检查 mic_layouts.yaml。"
            )

        pairs = build_mic_pairs(actual_num_mics)

        max_tau_samples = compute_max_tau_samples(
            mic_positions=mic_positions,
            pairs=pairs,
            fs=FS,
            sound_speed=SPEED_OF_SOUND,
        )

        baseline_weights = compute_baseline_weights(
            mic_positions=mic_positions,
            pairs=pairs,
        )

        print(f"[INFO] 麦克风对数量：{len(pairs)}")
        print(f"[INFO] 最大 TDOA 搜索范围：±{max_tau_samples} samples")

        print("[INFO] 预计算网格点理论 TDOA ...")
        grid_tdoa_cache = precompute_grid_tdoa_samples(
            grid_points=grid_points,
            mic_positions=mic_positions,
            pairs=pairs,
            fs=FS,
            sound_speed=SPEED_OF_SOUND,
        )

        for idx, row in source_positions_df.iterrows():
            scene_id = str(row["scene_id"])
            source_id = str(row["source_id"])

            true_position = np.array(
                [row["true_x"], row["true_y"], row["true_z"]],
                dtype=np.float64,
            )

            noise_seed = int(row["seed"]) + expected_num_mics * 10000

            multichannel = simulate_reverberant_noisy_scene(
                room_dim=room_dim,
                source_signal=source_signal,
                source_position=true_position,
                mic_positions=mic_positions,
                fs=FS,
                absorption=absorption,
                max_order=max_order_used,
                snr_db=BASELINE_SNR_DB,
                noise_seed=noise_seed,
            )

            pred_position, peak_score, _ = srp_phat_localize_one_scene(
                multichannel=multichannel,
                grid_points=grid_points,
                grid_shape=grid_shape,
                pairs=pairs,
                grid_tdoa_cache=grid_tdoa_cache,
                baseline_weights=baseline_weights,
                interp=INTERP,
                max_tau_samples=max_tau_samples,
            )

            error_cm = float(np.linalg.norm(pred_position[:2] - true_position[:2]) * 100.0)

            result_row = {
                "scene_id": scene_id,
                "source_id": source_id,
                "experiment_id": "E1",
                "variable": "mic_count",
                "value": int(expected_num_mics),

                "source_count": 1,
                "mic_layout": layout_name,
                "num_mics": int(actual_num_mics),
                "num_pairs": int(len(pairs)),

                "rt60": BASELINE_RT60_SEC,
                "snr": BASELINE_SNR_DB,
                "scene_type": "reverberant_single_source_rt60_0.30_snr_20db",

                "grid_spacing_m": GRID_SPACING_M,
                "search_plane_z": SEARCH_PLANE_Z,
                "source_plane_z": SOURCE_Z,

                "true_x": float(true_position[0]),
                "true_y": float(true_position[1]),
                "true_z": float(true_position[2]),

                "pred_x": float(pred_position[0]),
                "pred_y": float(pred_position[1]),
                "pred_z": float(pred_position[2]),

                "error_cm": error_cm,
                "hit_10cm": bool(error_cm <= 10.0),
                "hit_20cm": bool(error_cm <= 20.0),

                "score": float(peak_score),
                "srp_peak_score": float(peak_score),

                "matched": bool(error_cm <= 10.0),
                "seed": int(row["seed"]),
                "noise_seed": int(noise_seed),
            }

            all_rows.append(result_row)

            if (idx + 1) % 10 == 0:
                print(
                    f"[RUN] {layout_name} | {idx + 1:03d}/{NUM_SCENES_PER_GROUP} "
                    f"| error={error_cm:.3f} cm"
                )

        temp_df = pd.DataFrame(all_rows)
        temp_df.to_csv(SINGLE_SOURCE_CSV, index=False, encoding="utf-8-sig")
        print(f"[INFO] 已临时保存：{SINGLE_SOURCE_CSV}")

    print("\n[INFO] 全部 E1 场景定位完成，开始统计指标 ...")

    result_df = pd.DataFrame(all_rows)
    result_df = result_df.sort_values(["num_mics", "scene_id"]).reset_index(drop=True)
    result_df.to_csv(SINGLE_SOURCE_CSV, index=False, encoding="utf-8-sig")

    localization_cols = [
        "scene_id",
        "source_id",
        "experiment_id",
        "variable",
        "value",
        "source_count",
        "mic_layout",
        "num_mics",
        "rt60",
        "snr",
        "true_x",
        "true_y",
        "true_z",
        "pred_x",
        "pred_y",
        "pred_z",
        "error_cm",
        "score",
        "matched",
        "seed",
    ]

    result_df[localization_cols].to_csv(
        LOCALIZATION_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_results(result_df)
    summary_df.to_csv(EXPERIMENT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("[INFO] 绘制误差 CDF 图 ...")
    plot_error_cdf(result_df, CDF_PNG)

    print("[INFO] 绘制空间误差图 ...")
    plot_spatial_error(result_df, SPATIAL_ERROR_PNG)

    baseline_row = summary_df[summary_df["mic_layout"] == BASELINE_LAYOUT]

    if baseline_row.empty:
        baseline_pass = False
        baseline_mean = None
    else:
        baseline_mean = float(baseline_row.iloc[0]["mean_error_cm"])
        baseline_pass = baseline_mean <= BASELINE_MEAN_ERROR_THRESHOLD_CM

    print("[INFO] 写入 YAML 配置和 Markdown 报告 ...")

    scene_config = {
        "task": "week2_day3_e1_single_source_batch_localization",
        "script": "scripts/13_single_source_batch.py",
        "experiment": "E1_single_source_mic_count",
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "cage_yaml": str(CAGE_YAML.relative_to(PROJECT_ROOT)),
        "mic_layouts_yaml": str(MIC_LAYOUTS_YAML.relative_to(PROJECT_ROOT)),
        "cage_dimensions": cage_dims,

        "coordinate_system": {
            "x": "length",
            "y": "depth",
            "z": "height",
        },

        "fs": FS,
        "speed_of_sound": SPEED_OF_SOUND,
        "source_plane_z": SOURCE_Z,
        "search_plane_z": SEARCH_PLANE_Z,
        "rt60_sec": BASELINE_RT60_SEC,
        "snr_db": BASELINE_SNR_DB,
        "grid_spacing_m": GRID_SPACING_M,
        "grid_margin_x": GRID_MARGIN_X,
        "grid_margin_y": GRID_MARGIN_Y,
        "source_margin_x": SOURCE_MARGIN_X,
        "source_margin_y": SOURCE_MARGIN_Y,
        "num_scenes_per_group": NUM_SCENES_PER_GROUP,
        "total_runs": int(len(result_df)),

        "robust_srp_phat": {
            "interp": INTERP,
            "analysis_duration_sec": ANALYSIS_DURATION_SEC,
            "tdoa_score_window_samples": TDOA_SCORE_WINDOW_SAMPLES,
            "smooth_passes": SMOOTH_PASSES,
            "use_positive_normalized_gcc": True,
            "use_pair_reliability_weight": True,
            "use_baseline_weight": True,
        },

        "mic_layouts": [
            {"layout": name, "num_mics": n}
            for name, n in E1_LAYOUTS
        ],

        "pyroomacoustics": {
            "absorption": absorption,
            "max_order_raw": max_order_raw,
            "max_order_used": max_order_used,
            "max_order_cap": MAX_ORDER_CAP,
        },

        "metrics": [
            "mean_error_cm",
            "std_error_cm",
            "median_error_cm",
            "p90_error_cm",
            "hit_rate_10cm",
            "hit_rate_20cm",
            "ci95_error_cm",
        ],

        "baseline_check": {
            "baseline_layout": BASELINE_LAYOUT,
            "mean_error_threshold_cm": BASELINE_MEAN_ERROR_THRESHOLD_CM,
            "baseline_mean_error_cm": baseline_mean,
            "pass": bool(baseline_pass),
        },

        "outputs": {
            "single_source_csv": str(SINGLE_SOURCE_CSV.relative_to(PROJECT_ROOT)),
            "localization_results_csv": str(LOCALIZATION_RESULTS_CSV.relative_to(PROJECT_ROOT)),
            "experiment_summary_csv": str(EXPERIMENT_SUMMARY_CSV.relative_to(PROJECT_ROOT)),
            "scene_positions_csv": str(SCENE_POSITIONS_CSV.relative_to(PROJECT_ROOT)),
            "cdf_png": str(CDF_PNG.relative_to(PROJECT_ROOT)),
            "spatial_error_png": str(SPATIAL_ERROR_PNG.relative_to(PROJECT_ROOT)),
            "report_md": str(REPORT_MD.relative_to(PROJECT_ROOT)),
        },

        "simulation_boundary": (
            "几何声学仿真不精确表达金属笼条、鸡体遮挡、散射、衍射、复杂设备噪声和独立 USB 麦克风时钟漂移。"
            "source_id 为场景内临时编号，不代表固定蛋鸡个体。"
        ),
    }
    save_yaml(scene_config, SCENE_YAML)

    write_report(
        report_path=REPORT_MD,
        summary_df=summary_df,
        cage_dims=cage_dims,
        total_rows=len(result_df),
        baseline_pass=baseline_pass,
        absorption=absorption,
        max_order_used=max_order_used,
        max_order_raw=max_order_raw,
    )

    elapsed_sec = time.time() - start_time

    print("\n" + "=" * 80)
    print("[RESULT] 第 2 周 Day3：E1 单源批量定位完成")
    print(f"[RESULT] 总定位次数：{len(result_df)}")
    print(f"[RESULT] 输出 single_source.csv：{SINGLE_SOURCE_CSV}")
    print(f"[RESULT] 输出 localization_results.csv：{LOCALIZATION_RESULTS_CSV}")
    print(f"[RESULT] 输出 experiment_summary.csv：{EXPERIMENT_SUMMARY_CSV}")

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))

    if baseline_mean is None:
        print(f"\n[FAIL] 未找到基准配置 {BASELINE_LAYOUT}，无法判断 Day3 验收。")
    else:
        print(f"\n[CHECK] 基准配置 {BASELINE_LAYOUT} 平均误差：{baseline_mean:.3f} cm")
        print(f"[CHECK] 验收阈值：<= {BASELINE_MEAN_ERROR_THRESHOLD_CM:.1f} cm")
        print(f"[CHECK] 是否通过：{baseline_pass}")

        if baseline_pass:
            print("[PASS] 第 2 周 Day3 验收通过：基准配置 mic_8 平均误差满足 <= 10 cm。")
        else:
            print("[FAIL] 第 2 周 Day3 验收未通过：基准配置 mic_8 平均误差超过 10 cm。")

    print(f"[TIME] 总耗时：{elapsed_sec:.2f} 秒")


if __name__ == "__main__":
    main()