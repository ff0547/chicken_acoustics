# -*- coding: utf-8 -*-
"""
第2周 Day5：定位消融与阶段总结

任务：
1. 比较麦克风数量、布局、源间距、SNR 和 RT60；
2. 每组默认 100 场景，可用 --n-scenes 50 降低耗时；
3. 冻结默认算法参数，只改变实验变量；
4. 输出 localization_ablation.csv、4 幅图、week2.md。

实验设计：
E1：单源定位，麦克风数量 4/6/8/12
E2：单源定位，布局 四角/两侧/单边/集中
E3：单源定位，SNR 0/5/10/20/30 dB
E4：双源定位，源间距 10/20/30/50/80 cm
E5：单源定位，RT60 0.1/0.3/0.5/0.7 s

运行：
    cd D:\\project\\chicken_acoustics
    python scripts\\15_localization_ablation.py --n-scenes 3

正式运行：
    python scripts\\15_localization_ablation.py --n-scenes 100
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ============================================================
# 0. 路径与动态加载
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DAY3_SCRIPT = PROJECT_ROOT / "scripts" / "13_single_source_batch.py"
DAY4_SCRIPT = PROJECT_ROOT / "scripts" / "14_dual_source_multipeak.py"

OUTPUT_DIR = PROJECT_ROOT / "results" / "week2" / "day5"
SCENE_CSV = OUTPUT_DIR / "localization_ablation_scenes.csv"
SUMMARY_CSV = OUTPUT_DIR / "localization_ablation.csv"
CONFIG_YAML = OUTPUT_DIR / "localization_ablation_config.yaml"
WEEK2_MD = PROJECT_ROOT / "results" / "week2" / "week2.md"

FIG_E1 = OUTPUT_DIR / "fig_e1_num_mics.png"
FIG_E2 = OUTPUT_DIR / "fig_e2_layout.png"
FIG_E3 = OUTPUT_DIR / "fig_e3_noise_reverb.png"
FIG_E4 = OUTPUT_DIR / "fig_e4_dual_distance.png"


def load_module(script_path: Path, module_name: str):
    """动态加载脚本模块，避免复制 Day3/Day4 的核心算法。"""
    if not script_path.exists():
        raise FileNotFoundError(f"找不到脚本：{script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{script_path}")

    module = importlib.util.module_from_spec(spec)

    # 关键修复：
    # dataclass 装饰器会通过 sys.modules[cls.__module__] 找模块命名空间。
    # 动态加载含 @dataclass 的脚本时，必须先注册到 sys.modules，再 exec_module。
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


day3 = load_module(DAY3_SCRIPT, "day3_single_source_batch")
day4 = load_module(DAY4_SCRIPT, "day4_dual_source_multipeak")


# ============================================================
# 1. Day5 默认实验参数
# ============================================================

DEFAULT_LAYOUT = "mic_8"
DEFAULT_SNR_DB = 20.0
DEFAULT_RT60_SEC = 0.30
DEFAULT_GRID_SPACING_M = 0.02
DEFAULT_HIT_RADIUS_M = 0.10

SOURCE_Z = 0.35
SEARCH_Z = 0.35

RANDOM_SEED = 20260629

# E1：麦数消融
E1_MIC_LAYOUTS = [
    ("mic_4", 4),
    ("mic_6", 6),
    ("mic_8", 8),
    ("mic_12", 12),
]

# E2：布局消融
# 注意：这里使用你 mic_layouts.yaml 里的真实命名
E2_LAYOUTS = [
    ("layout_corners_8", "四角布局"),
    ("layout_sides_8", "两侧布局"),
    ("layout_single_side_8", "单边布局"),
    ("layout_compact_8", "集中布局"),
]

# E3：SNR 消融
E3_SNRS_DB = [0.0, 5.0, 10.0, 20.0, 30.0]

# E4：双源源间距消融
E4_DISTANCES_CM = [10.0, 20.0, 30.0, 50.0, 80.0]

# E5：RT60 消融
E5_RT60S_SEC = [0.10, 0.30, 0.50, 0.70]


# ============================================================
# 2. 数据结构与工具函数
# ============================================================

@dataclass
class PreparedLayout:
    """某个布局 + 某个 RT60 下的预计算对象。"""

    cage_dims: Dict[str, float]
    room_dim: np.ndarray
    mic_positions: np.ndarray
    num_mics: int
    x_values: np.ndarray
    y_values: np.ndarray
    grid_points: np.ndarray
    grid_shape: Tuple[int, int]
    pairs: List[Tuple[int, int]]
    grid_tdoa_cache: Dict[Tuple[int, int], np.ndarray]
    baseline_weights: Dict[Tuple[int, int], float]
    absorption: float
    max_order_used: int
    max_order_raw: int
    max_tau_samples: int


def reset_output_dir(path: Path) -> None:
    """清空并重建 Day5 输出目录。"""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """保存 YAML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def safe_std(values: np.ndarray) -> float:
    """样本标准差；长度不足时返回 0。"""
    values = np.asarray(values, dtype=np.float64)

    if len(values) <= 1:
        return 0.0

    return float(np.std(values, ddof=1))


def safe_mean(values: np.ndarray) -> float:
    """忽略 NaN 的均值。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_p90(values: np.ndarray) -> float:
    """忽略 NaN 的 P90。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, 90))



def compute_peak_ratio(
    score_map: np.ndarray,
    min_separation_m: float = 0.08,
    grid_spacing_m: float = DEFAULT_GRID_SPACING_M,
    eps: float = 1e-12,
) -> float:
    """
    计算 SRP 空间谱峰值比。

    定义：
        peak_ratio = 第一主峰值 / 第二主峰值

    说明：
        这里的第二峰不是简单取相邻网格点，而是要求与第一峰至少间隔
        min_separation_m，避免把同一主峰附近的相邻格点误当成第二峰。

    含义：
        peak_ratio 越大，说明主峰越突出，定位越稳定；
        peak_ratio 越接近 1，说明存在强度接近的伪峰或混响峰。
    """
    if score_map is None:
        return np.nan

    score = np.asarray(score_map, dtype=np.float64)

    if score.ndim != 2:
        score = score.reshape(-1, 1)

    finite_mask = np.isfinite(score)

    if finite_mask.sum() < 2:
        return np.nan

    safe_score = np.where(finite_mask, score, -np.inf)
    flat_order = np.argsort(safe_score.reshape(-1))[::-1]

    first_flat = int(flat_order[0])
    first_y, first_x = np.unravel_index(first_flat, safe_score.shape)
    first_peak = float(safe_score[first_y, first_x])

    min_sep_cells = max(1, int(round(float(min_separation_m) / float(grid_spacing_m))))

    second_peak = np.nan

    for flat_idx in flat_order[1:]:
        y, x = np.unravel_index(int(flat_idx), safe_score.shape)

        if not np.isfinite(safe_score[y, x]):
            continue

        dist_cells = math.sqrt((float(y) - float(first_y)) ** 2 + (float(x) - float(first_x)) ** 2)

        if dist_cells >= min_sep_cells:
            second_peak = float(safe_score[y, x])
            break

    if not np.isfinite(second_peak):
        return np.nan

    if abs(second_peak) <= eps:
        return np.nan

    return float(first_peak / (second_peak + eps))


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
                if np.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def get_room_dim(cage_dims: Dict[str, float]) -> np.ndarray:
    """由 cage.yaml 的尺寸生成 Pyroomacoustics 房间尺寸。"""
    return np.array(
        [
            cage_dims["length_x"],
            cage_dims["width_y"],
            cage_dims["height_z"],
        ],
        dtype=np.float64,
    )


def compute_material_and_order(room_dim: np.ndarray, rt60_sec: float) -> Tuple[float, int, int]:
    """
    根据 RT60 计算吸声系数和 max_order。

    说明：
    - RT60=0 时使用 max_order=0，近似无混响。
    - RT60=0.30 时与 Day3/Day4 默认一致。
    """
    rt60_sec = float(rt60_sec)

    if rt60_sec <= 1e-9:
        return 1.0, 0, 0

    try:
        absorption, max_order_raw = day3.pra.inverse_sabine(rt60_sec, room_dim)
        max_order_raw = int(max_order_raw)
    except Exception:
        if abs(rt60_sec - float(day3.BASELINE_RT60_SEC)) < 1e-9:
            absorption, max_order_used, max_order_raw = day3.compute_room_material_and_order(room_dim)
            return float(absorption), int(max_order_used), int(max_order_raw)

        raise

    max_order_cap = int(getattr(day3, "MAX_ORDER_CAP", max_order_raw))
    max_order_used = min(max_order_raw, max_order_cap)

    return float(absorption), int(max_order_used), int(max_order_raw)


def prepare_layout(
    layout_name: str,
    rt60_sec: float,
    search_z: float,
    grid_spacing_m: float,
) -> PreparedLayout:
    """加载布局，并预计算网格 TDOA、麦克风对、基线权重等。"""
    cage_cfg = day3.load_yaml(day3.CAGE_YAML)
    mic_cfg = day3.load_yaml(day3.MIC_LAYOUTS_YAML)

    cage_dims = day3.get_cage_dimensions(cage_cfg)
    room_dim = get_room_dim(cage_dims)

    mic_df = day3.extract_mic_layout(mic_cfg, layout_name)
    mic_positions = mic_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    num_mics = int(len(mic_df))

    x_values, y_values, grid_points = day3.make_grid(
        length_x=cage_dims["length_x"],
        width_y=cage_dims["width_y"],
        spacing=grid_spacing_m,
        margin_x=day3.GRID_MARGIN_X,
        margin_y=day3.GRID_MARGIN_Y,
        z=search_z,
    )

    grid_shape = (len(y_values), len(x_values))

    pairs = day3.build_mic_pairs(num_mics)

    max_tau_samples = day3.compute_max_tau_samples(
        mic_positions=mic_positions,
        pairs=pairs,
        fs=day3.FS,
        sound_speed=day3.SPEED_OF_SOUND,
    )

    baseline_weights = day3.compute_baseline_weights(
        mic_positions=mic_positions,
        pairs=pairs,
    )

    grid_tdoa_cache = day3.precompute_grid_tdoa_samples(
        grid_points=grid_points,
        mic_positions=mic_positions,
        pairs=pairs,
        fs=day3.FS,
        sound_speed=day3.SPEED_OF_SOUND,
    )

    absorption, max_order_used, max_order_raw = compute_material_and_order(
        room_dim=room_dim,
        rt60_sec=rt60_sec,
    )

    return PreparedLayout(
        cage_dims=cage_dims,
        room_dim=room_dim,
        mic_positions=mic_positions,
        num_mics=num_mics,
        x_values=x_values,
        y_values=y_values,
        grid_points=grid_points,
        grid_shape=grid_shape,
        pairs=pairs,
        grid_tdoa_cache=grid_tdoa_cache,
        baseline_weights=baseline_weights,
        absorption=absorption,
        max_order_used=max_order_used,
        max_order_raw=max_order_raw,
        max_tau_samples=max_tau_samples,
    )


def check_layouts_exist() -> None:
    """检查 Day5 需要的布局是否都存在。"""
    required_layouts = [name for name, _ in E1_MIC_LAYOUTS]
    required_layouts += [name for name, _ in E2_LAYOUTS]

    mic_cfg = day3.load_yaml(day3.MIC_LAYOUTS_YAML)

    for layout_name in required_layouts:
        mic_df = day3.extract_mic_layout(mic_cfg, layout_name)

        if len(mic_df) == 0:
            raise ValueError(f"布局为空：{layout_name}")

    print("[CHECK] Day5 所需麦克风布局均已存在。")


# ============================================================
# 3. 单源场景生成与定位
# ============================================================

def sample_single_source_position(
    cage_dims: Dict[str, float],
    rng: np.random.Generator,
    source_z: float,
) -> np.ndarray:
    """随机采样一个单源位置。"""
    x_min = float(day3.SOURCE_MARGIN_X)
    x_max = float(cage_dims["length_x"] - day3.SOURCE_MARGIN_X)

    y_min = float(day3.SOURCE_MARGIN_Y)
    y_max = float(cage_dims["width_y"] - day3.SOURCE_MARGIN_Y)

    return np.array(
        [
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
            source_z,
        ],
        dtype=np.float64,
    )


def simulate_single_source_scene(
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
    """使用 Pyroomacoustics 生成单源混响加噪多通道信号。"""
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

    room.add_source(source_position, signal=source_signal)

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


def run_one_single_scene(
    scene_index: int,
    prepared: PreparedLayout,
    layout_name: str,
    experiment_id: str,
    variable_name: str,
    condition_name: str,
    snr_db: float,
    rt60_sec: float,
    seed_base: int,
    source_z: float,
    hit_radius_m: float,
) -> Dict[str, Any]:
    """运行一个单源场景，并返回逐场景结果。"""
    scene_seed = int(seed_base + scene_index)
    noise_seed = scene_seed + 500000
    rng = np.random.default_rng(scene_seed)

    source_position = sample_single_source_position(
        cage_dims=prepared.cage_dims,
        rng=rng,
        source_z=source_z,
    )

    source_signal = day3.make_probe_signal(
        fs=day3.FS,
        duration_sec=day3.PROBE_DURATION_SEC,
        seed=scene_seed + 1000,
    )

    source_signal = source_signal.astype(np.float64)
    source_signal = source_signal / (np.max(np.abs(source_signal)) + 1e-12)

    t0 = time.perf_counter()

    try:
        multichannel = simulate_single_source_scene(
            room_dim=prepared.room_dim,
            source_signal=source_signal,
            source_position=source_position,
            mic_positions=prepared.mic_positions,
            fs=day3.FS,
            absorption=prepared.absorption,
            max_order=prepared.max_order_used,
            snr_db=snr_db,
            noise_seed=noise_seed,
        )

        pred_position, peak_score, score_map = day3.srp_phat_localize_one_scene(
            multichannel=multichannel,
            grid_points=prepared.grid_points,
            grid_shape=prepared.grid_shape,
            pairs=prepared.pairs,
            grid_tdoa_cache=prepared.grid_tdoa_cache,
            baseline_weights=prepared.baseline_weights,
            interp=day3.INTERP,
            max_tau_samples=prepared.max_tau_samples,
        )

        peak_ratio = compute_peak_ratio(score_map)

        error_m = float(np.linalg.norm(pred_position[:2] - source_position[:2]))
        hit = bool(error_m <= hit_radius_m)
        failed = False

    except Exception as exc:
        pred_position = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        peak_score = np.nan
        peak_ratio = np.nan
        error_m = np.nan
        hit = False
        failed = True
        print(f"[WARN] 单源场景失败：{experiment_id} {condition_name} scene={scene_index + 1}, error={exc}")

    runtime_s = time.perf_counter() - t0

    return {
        "scene_id": f"{experiment_id}_{condition_name}_scene_{scene_index + 1:04d}",
        "experiment_id": experiment_id,
        "task_type": "single_source",
        "variable_name": variable_name,
        "condition_name": condition_name,

        "num_sources": 1,
        "mic_layout": layout_name,
        "num_mics": prepared.num_mics,
        "num_pairs": len(prepared.pairs),

        "snr_db": float(snr_db),
        "rt60_sec": float(rt60_sec),
        "grid_spacing_m": float(DEFAULT_GRID_SPACING_M),
        "hit_radius_m": float(hit_radius_m),

        "true_x": float(source_position[0]),
        "true_y": float(source_position[1]),
        "true_z": float(source_position[2]),

        "pred_x": float(pred_position[0]),
        "pred_y": float(pred_position[1]),
        "pred_z": float(pred_position[2]),

        "peak_score": float(peak_score) if np.isfinite(peak_score) else np.nan,
        "peak_ratio": float(peak_ratio) if np.isfinite(peak_ratio) else np.nan,
        "error_m": float(error_m) if np.isfinite(error_m) else np.nan,
        "error_cm": float(error_m * 100.0) if np.isfinite(error_m) else np.nan,
        "hit_10cm": bool(hit),
        "miss": bool(not hit),

        "runtime_s": float(runtime_s),
        "failed": bool(failed),

        "seed": int(scene_seed),
        "noise_seed": int(noise_seed),
        "absorption": float(prepared.absorption),
        "max_order_used": int(prepared.max_order_used),
    }


def run_single_condition(
    experiment_id: str,
    variable_name: str,
    condition_name: str,
    layout_name: str,
    n_scenes: int,
    seed_base: int,
    snr_db: float,
    rt60_sec: float,
    source_z: float,
    search_z: float,
    grid_spacing_m: float,
    hit_radius_m: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """运行一个单源消融条件。"""
    print("\n" + "-" * 80)
    print(f"[RUN] {experiment_id} | {condition_name} | 单源定位")
    print("-" * 80)

    prepared = prepare_layout(
        layout_name=layout_name,
        rt60_sec=rt60_sec,
        search_z=search_z,
        grid_spacing_m=grid_spacing_m,
    )

    rows = []

    for scene_index in range(n_scenes):
        row = run_one_single_scene(
            scene_index=scene_index,
            prepared=prepared,
            layout_name=layout_name,
            experiment_id=experiment_id,
            variable_name=variable_name,
            condition_name=condition_name,
            snr_db=snr_db,
            rt60_sec=rt60_sec,
            seed_base=seed_base,
            source_z=source_z,
            hit_radius_m=hit_radius_m,
        )

        rows.append(row)

        if (scene_index + 1) % max(1, n_scenes // 10) == 0 or scene_index == 0:
            temp = pd.DataFrame(rows)
            hit_rate = float(temp["hit_10cm"].astype(bool).mean())
            mean_error = safe_mean(temp["error_m"].to_numpy(dtype=np.float64))

            print(
                f"[RUN] {scene_index + 1:03d}/{n_scenes} "
                f"| hit_rate={hit_rate:.3f} "
                f"| mean_error={mean_error * 100:.2f} cm",
                flush=True,
            )

    scene_df = pd.DataFrame(rows)
    summary = summarize_single_condition(scene_df)

    return scene_df, summary


# ============================================================
# 4. 双源源间距消融
# ============================================================

def make_target_distance_sampler(target_distance_m: float, tolerance_m: float = 0.015):
    """
    为 E4 构造目标源间距采样器。

    Day4 原函数是 min_distance 采样；
    Day5 做源间距消融时，改为在 target_distance_m 附近采样，
    避免 >=10cm 组混入大量远距离样本。
    """

    def sampler(
        cage_dims: Dict[str, float],
        rng: np.random.Generator,
        min_distance_m: float,
        source_z: float,
        max_trials: int = 20000,
    ) -> np.ndarray:
        x_min = float(day3.SOURCE_MARGIN_X)
        x_max = float(cage_dims["length_x"] - day3.SOURCE_MARGIN_X)

        y_min = float(day3.SOURCE_MARGIN_Y)
        y_max = float(cage_dims["width_y"] - day3.SOURCE_MARGIN_Y)

        d_low = max(0.02, target_distance_m - tolerance_m)
        d_high = target_distance_m + tolerance_m

        for _ in range(max_trials):
            p1 = np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    source_z,
                ],
                dtype=np.float64,
            )

            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            distance = float(rng.uniform(d_low, d_high))

            p2 = np.array(
                [
                    p1[0] + distance * math.cos(theta),
                    p1[1] + distance * math.sin(theta),
                    source_z,
                ],
                dtype=np.float64,
            )

            if x_min <= p2[0] <= x_max and y_min <= p2[1] <= y_max:
                return np.stack([p1, p2], axis=0)

        raise RuntimeError(f"无法采样到目标源间距约 {target_distance_m:.3f} m 的双源位置。")

    return sampler


def make_day4_args(
    n_scenes: int,
    seed: int,
    layout_name: str,
    min_source_distance_m: float,
    snr_db: float,
    rt60_sec: float,
    grid_spacing_m: float,
    source_z: float,
    search_z: float,
    hit_radius_m: float,
) -> SimpleNamespace:
    """构造 Day4 run_one_scene() 需要的参数对象。"""
    return SimpleNamespace(
        n_scenes=int(n_scenes),
        seed=int(seed),

        layout_name=layout_name,

        source_z=float(source_z),
        search_z=float(search_z),

        rt60_sec=float(rt60_sec),
        snr_db=float(snr_db),
        grid_spacing_m=float(grid_spacing_m),

        min_source_distance_m=float(min_source_distance_m),
        hit_radius_m=float(hit_radius_m),

        total_duration_sec=1.10,

        frame_sec=0.24,
        hop_sec=0.035,
        max_active_frames=8,
        energy_threshold_rel=0.06,
        frames_per_event=2,
        min_event_gap_sec=0.28,

        density_sigma_m=0.045,
        cluster_sigma_m=0.040,
        cluster_trim_radius_m=0.12,

        density_weight=0.15,
        cluster_prior_weight=0.85,
        hybrid_score_gamma=1.20,

        peak_threshold_rel=0.03,
        nms_radius_m=0.08,
        min_pred_pair_distance_m=0.18,
        max_peak_candidates=30,

        max_failure_plots=0,
        max_tau_samples=None,
    )


def run_dual_condition(
    experiment_id: str,
    variable_name: str,
    condition_name: str,
    target_distance_cm: float,
    n_scenes: int,
    seed_base: int,
    layout_name: str,
    snr_db: float,
    rt60_sec: float,
    source_z: float,
    search_z: float,
    grid_spacing_m: float,
    hit_radius_m: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """运行 E4 双源源间距消融条件。"""
    print("\n" + "-" * 80)
    print(f"[RUN] {experiment_id} | {condition_name} | 双源定位")
    print("-" * 80)

    target_distance_m = float(target_distance_cm) / 100.0

    prepared = prepare_layout(
        layout_name=layout_name,
        rt60_sec=rt60_sec,
        search_z=search_z,
        grid_spacing_m=grid_spacing_m,
    )

    args = make_day4_args(
        n_scenes=n_scenes,
        seed=seed_base,
        layout_name=layout_name,
        min_source_distance_m=target_distance_m,
        snr_db=snr_db,
        rt60_sec=rt60_sec,
        grid_spacing_m=grid_spacing_m,
        source_z=source_z,
        search_z=search_z,
        hit_radius_m=hit_radius_m,
    )

    args.max_tau_samples = prepared.max_tau_samples

    # 将 Day4 的全局示例图输出临时改到 Day5，避免污染 Day4 结果。
    old_sampler = day4.sample_dual_source_positions
    old_example = day4.EXAMPLE_PNG
    old_frame = day4.FRAME_PNG
    old_failure_dir = day4.FAILURE_DIR

    condition_dir = OUTPUT_DIR / "e4_examples" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    day4.sample_dual_source_positions = make_target_distance_sampler(target_distance_m)
    day4.EXAMPLE_PNG = condition_dir / "example_srp_peaks.png"
    day4.FRAME_PNG = condition_dir / "example_frame_positions.png"
    day4.FAILURE_DIR = condition_dir / "failure_cases"
    day4.FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    try:
        for scene_index in range(n_scenes):
            t0 = time.perf_counter()

            row = day4.run_one_scene(
                scene_index=scene_index,
                cage_dims=prepared.cage_dims,
                room_dim=prepared.room_dim,
                mic_positions=prepared.mic_positions,
                x_values=prepared.x_values,
                y_values=prepared.y_values,
                grid_points=prepared.grid_points,
                grid_shape=prepared.grid_shape,
                pairs=prepared.pairs,
                grid_tdoa_cache=prepared.grid_tdoa_cache,
                baseline_weights=prepared.baseline_weights,
                absorption=prepared.absorption,
                max_order_used=prepared.max_order_used,
                args=args,
            )

            runtime_s = time.perf_counter() - t0

            row["scene_id"] = f"{experiment_id}_{condition_name}_scene_{scene_index + 1:04d}"
            row["experiment_id"] = experiment_id
            row["task_type"] = "dual_source"
            row["variable_name"] = variable_name
            row["condition_name"] = condition_name
            row["target_source_distance_cm"] = float(target_distance_cm)
            row["runtime_s"] = float(runtime_s)
            row["hit_radius_m"] = float(hit_radius_m)
            row["snr_db"] = float(snr_db)
            row["rt60_sec"] = float(rt60_sec)

            rows.append(row)

            if (scene_index + 1) % max(1, n_scenes // 10) == 0 or scene_index == 0:
                temp = pd.DataFrame(rows)
                dual_hit_rate = float(temp["both_hit"].astype(bool).mean())
                mean_error_cm = safe_mean(temp["mean_error_cm"].to_numpy(dtype=np.float64))

                print(
                    f"[RUN] {scene_index + 1:03d}/{n_scenes} "
                    f"| dual_hit_rate={dual_hit_rate:.3f} "
                    f"| mean_error={mean_error_cm:.2f} cm",
                    flush=True,
                )

    finally:
        day4.sample_dual_source_positions = old_sampler
        day4.EXAMPLE_PNG = old_example
        day4.FRAME_PNG = old_frame
        day4.FAILURE_DIR = old_failure_dir

    scene_df = pd.DataFrame(rows)
    summary = summarize_dual_condition(scene_df)

    return scene_df, summary


# ============================================================
# 5. 汇总统计
# ============================================================

def summarize_single_condition(scene_df: pd.DataFrame) -> Dict[str, Any]:
    """汇总单源定位条件。"""
    n = int(len(scene_df))

    errors = scene_df["error_m"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    hits = scene_df["hit_10cm"].astype(bool).to_numpy(dtype=bool)
    misses = scene_df["miss"].astype(bool).to_numpy(dtype=bool)
    runtimes = scene_df["runtime_s"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)

    if "peak_ratio" in scene_df.columns:
        peak_ratios = scene_df["peak_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    else:
        peak_ratios = np.array([], dtype=np.float64)

    hit_values = hits.astype(np.float64)
    miss_values = misses.astype(np.float64)

    row0 = scene_df.iloc[0]

    return {
        "experiment_id": str(row0["experiment_id"]),
        "task_type": "single_source",
        "variable_name": str(row0["variable_name"]),
        "condition_name": str(row0["condition_name"]),

        "num_scenes": n,
        "num_sources": 1,
        "mic_layout": str(row0["mic_layout"]),
        "num_mics": int(row0["num_mics"]),
        "source_distance_cm": np.nan,

        "snr_db": float(row0["snr_db"]),
        "rt60_sec": float(row0["rt60_sec"]),
        "grid_spacing_m": float(row0["grid_spacing_m"]),
        "hit_radius_m": float(row0["hit_radius_m"]),

        "mae_m": safe_mean(errors),
        "p90_error_m": safe_p90(errors),
        "error_mean_m": safe_mean(errors),
        "error_std_m": safe_std(errors),

        "peak_ratio_mean": safe_mean(peak_ratios),
        "peak_ratio_std": safe_std(peak_ratios),

        "hit_rate_10cm": float(np.mean(hit_values)) if len(hit_values) > 0 else np.nan,
        "hit_rate_10cm_std": safe_std(hit_values),
        "miss_rate": float(np.mean(miss_values)) if len(miss_values) > 0 else np.nan,
        "miss_rate_std": safe_std(miss_values),

        "dual_hit_rate": np.nan,
        "dual_hit_rate_std": np.nan,
        "false_alarm_rate": np.nan,
        "false_alarm_rate_std": np.nan,

        "runtime_mean_s": safe_mean(runtimes),
        "runtime_std_s": safe_std(runtimes),

        "failed_count": int(scene_df["failed"].astype(bool).sum()) if "failed" in scene_df.columns else 0,
        "seed": int(row0["seed"]),
    }


def summarize_dual_condition(scene_df: pd.DataFrame) -> Dict[str, Any]:
    """汇总双源定位条件。"""
    n = int(len(scene_df))
    row0 = scene_df.iloc[0]

    both_hit = scene_df["both_hit"].astype(bool).to_numpy(dtype=bool)
    both_hit_values = both_hit.astype(np.float64)

    total_true = float(scene_df["num_true_sources"].sum())
    total_pred = float(scene_df["num_pred_peaks"].sum())

    miss_rate = float(scene_df["miss_count"].sum() / total_true) if total_true > 0 else np.nan
    false_alarm_rate = float(scene_df["false_alarm_count"].sum() / total_pred) if total_pred > 0 else np.nan

    per_scene_miss_rate = (
        scene_df["miss_count"].to_numpy(dtype=np.float64)
        / np.maximum(scene_df["num_true_sources"].to_numpy(dtype=np.float64), 1.0)
    )

    per_scene_false_alarm_rate = (
        scene_df["false_alarm_count"].to_numpy(dtype=np.float64)
        / np.maximum(scene_df["num_pred_peaks"].to_numpy(dtype=np.float64), 1.0)
    )

    errors_m = (
        scene_df["mean_error_cm"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float64)
        / 100.0
    )

    runtimes = (
        scene_df["runtime_s"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    return {
        "experiment_id": str(row0["experiment_id"]),
        "task_type": "dual_source",
        "variable_name": str(row0["variable_name"]),
        "condition_name": str(row0["condition_name"]),

        "num_scenes": n,
        "num_sources": 2,
        "mic_layout": str(row0["mic_layout"]),
        "num_mics": int(row0["num_mics"]),
        "source_distance_cm": float(row0["target_source_distance_cm"]),

        "snr_db": float(row0["snr_db"]),
        "rt60_sec": float(row0["rt60_sec"]),
        "grid_spacing_m": float(row0["grid_spacing_m"]),
        "hit_radius_m": float(row0["hit_radius_m"]),

        "mae_m": safe_mean(errors_m),
        "p90_error_m": safe_p90(errors_m),
        "error_mean_m": safe_mean(errors_m),
        "error_std_m": safe_std(errors_m),

        "peak_ratio_mean": np.nan,
        "peak_ratio_std": np.nan,

        "hit_rate_10cm": np.nan,
        "hit_rate_10cm_std": np.nan,
        "miss_rate": miss_rate,
        "miss_rate_std": safe_std(per_scene_miss_rate),

        "dual_hit_rate": float(np.mean(both_hit_values)) if len(both_hit_values) > 0 else np.nan,
        "dual_hit_rate_std": safe_std(both_hit_values),
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_rate_std": safe_std(per_scene_false_alarm_rate),

        "runtime_mean_s": safe_mean(runtimes),
        "runtime_std_s": safe_std(runtimes),

        "failed_count": 0,
        "seed": int(row0["seed"]),
    }


def add_control_flag(summary_df: pd.DataFrame) -> pd.DataFrame:
    """标记每个消融变量的控制组。"""
    df = summary_df.copy()
    df["is_control"] = False

    df.loc[
        (df["experiment_id"] == "E1_num_mics") & (df["condition_name"] == "mic_8"),
        "is_control",
    ] = True

    df.loc[
        (df["experiment_id"] == "E2_layout") & (df["condition_name"] == "layout_corners_8"),
        "is_control",
    ] = True

    df.loc[
        (df["experiment_id"] == "E3_snr") & (df["condition_name"] == "snr_20db"),
        "is_control",
    ] = True

    df.loc[
        (df["experiment_id"] == "E4_dual_distance") & (df["condition_name"] == "dist_30cm"),
        "is_control",
    ] = True

    df.loc[
        (df["experiment_id"] == "E5_rt60") & (df["condition_name"] == "rt60_0.30s"),
        "is_control",
    ] = True

    return df


# ============================================================
# 6. 绘图
# ============================================================

def plot_errorbar(
    ax,
    x_labels: List[str],
    y_values: np.ndarray,
    y_err: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    """画均值±标准差误差棒。"""
    x = np.arange(len(x_labels))

    ax.errorbar(
        x,
        y_values,
        yerr=y_err,
        fmt="o-",
        capsize=4,
        linewidth=1.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def generate_figures(summary_df: pd.DataFrame) -> None:
    """
    生成 Day5 要求的 4 幅图。

    4 幅图覆盖 5 个消融变量：
    1. E1：麦克风数量——MAE/P90/10cm 命中率
    2. E2：布局——误差/运行时间
    3. E3+E5：SNR 与 RT60——误差、漏检率、峰值比
    4. E4：双源源间距——双源命中率、虚警率
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # E1：麦数消融。
    e1 = summary_df[summary_df["experiment_id"] == "E1_num_mics"].copy()

    if not e1.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        x_labels = e1["condition_name"].tolist()
        x = np.arange(len(x_labels))

        axes[0].errorbar(
            x,
            e1["mae_m"].to_numpy(dtype=np.float64) * 100.0,
            yerr=e1["error_std_m"].to_numpy(dtype=np.float64) * 100.0,
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axes[0].plot(x, e1["p90_error_m"].to_numpy(dtype=np.float64) * 100.0, "s--", linewidth=1.2)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[0].set_title("E1 Num Mics: Error")
        axes[0].set_ylabel("Error / cm")
        axes[0].grid(alpha=0.3)
        axes[0].legend(["MAE ± STD", "P90"])

        axes[1].errorbar(
            x,
            e1["hit_rate_10cm"].to_numpy(dtype=np.float64),
            yerr=e1["hit_rate_10cm_std"].to_numpy(dtype=np.float64),
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[1].set_title("E1 Num Mics: 10 cm Hit Rate")
        axes[1].set_ylabel("Hit Rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIG_E1, dpi=180)
        plt.close()

    # E2：布局消融。
    e2 = summary_df[summary_df["experiment_id"] == "E2_layout"].copy()

    if not e2.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        x_labels = e2["condition_name"].tolist()
        x = np.arange(len(x_labels))

        axes[0].errorbar(
            x,
            e2["mae_m"].to_numpy(dtype=np.float64) * 100.0,
            yerr=e2["error_std_m"].to_numpy(dtype=np.float64) * 100.0,
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[0].set_title("E2 Layout: Error")
        axes[0].set_ylabel("MAE / cm")
        axes[0].grid(alpha=0.3)

        axes[1].errorbar(
            x,
            e2["runtime_mean_s"].to_numpy(dtype=np.float64),
            yerr=e2["runtime_std_s"].to_numpy(dtype=np.float64),
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[1].set_title("E2 Layout: Runtime")
        axes[1].set_ylabel("Runtime / s")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIG_E2, dpi=180)
        plt.close()

    # E3 + E5：SNR 与 RT60 合并成一幅图。
    e3 = summary_df[summary_df["experiment_id"] == "E3_snr"].copy()
    e5 = summary_df[summary_df["experiment_id"] == "E5_rt60"].copy()

    if not e3.empty or not e5.empty:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        if not e3.empty:
            x_labels = e3["condition_name"].tolist()
            x = np.arange(len(x_labels))

            axes[0, 0].errorbar(
                x,
                e3["mae_m"].to_numpy(dtype=np.float64) * 100.0,
                yerr=e3["error_std_m"].to_numpy(dtype=np.float64) * 100.0,
                fmt="o-",
                capsize=4,
                linewidth=1.5,
            )
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(x_labels, rotation=20, ha="right")
            axes[0, 0].set_title("E3 SNR: Error")
            axes[0, 0].set_ylabel("MAE / cm")
            axes[0, 0].grid(alpha=0.3)

            axes[0, 1].errorbar(
                x,
                e3["miss_rate"].to_numpy(dtype=np.float64),
                yerr=e3["miss_rate_std"].to_numpy(dtype=np.float64),
                fmt="o-",
                capsize=4,
                linewidth=1.5,
            )
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(x_labels, rotation=20, ha="right")
            axes[0, 1].set_title("E3 SNR: Miss Rate")
            axes[0, 1].set_ylabel("Miss Rate")
            axes[0, 1].set_ylim(-0.05, 1.05)
            axes[0, 1].grid(alpha=0.3)
        else:
            axes[0, 0].axis("off")
            axes[0, 1].axis("off")

        if not e5.empty:
            x_labels = e5["condition_name"].tolist()
            x = np.arange(len(x_labels))

            axes[1, 0].errorbar(
                x,
                e5["mae_m"].to_numpy(dtype=np.float64) * 100.0,
                yerr=e5["error_std_m"].to_numpy(dtype=np.float64) * 100.0,
                fmt="o-",
                capsize=4,
                linewidth=1.5,
            )
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(x_labels, rotation=20, ha="right")
            axes[1, 0].set_title("E5 RT60: Error")
            axes[1, 0].set_ylabel("MAE / cm")
            axes[1, 0].grid(alpha=0.3)

            axes[1, 1].errorbar(
                x,
                e5["peak_ratio_mean"].to_numpy(dtype=np.float64),
                yerr=e5["peak_ratio_std"].to_numpy(dtype=np.float64),
                fmt="o-",
                capsize=4,
                linewidth=1.5,
            )
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(x_labels, rotation=20, ha="right")
            axes[1, 1].set_title("E5 RT60: Peak Ratio")
            axes[1, 1].set_ylabel("Peak Ratio")
            axes[1, 1].grid(alpha=0.3)
        else:
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(FIG_E3, dpi=180)
        plt.close()

    # E4：双源源间距消融。
    e4 = summary_df[summary_df["experiment_id"] == "E4_dual_distance"].copy()

    if not e4.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        x_labels = e4["condition_name"].tolist()
        x = np.arange(len(x_labels))

        axes[0].errorbar(
            x,
            e4["dual_hit_rate"].to_numpy(dtype=np.float64),
            yerr=e4["dual_hit_rate_std"].to_numpy(dtype=np.float64),
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[0].set_title("E4 Dual Distance: Dual Hit Rate")
        axes[0].set_ylabel("Dual Hit Rate")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(alpha=0.3)

        axes[1].errorbar(
            x,
            e4["false_alarm_rate"].to_numpy(dtype=np.float64),
            yerr=e4["false_alarm_rate_std"].to_numpy(dtype=np.float64),
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[1].set_title("E4 Dual Distance: False Alarm Rate")
        axes[1].set_ylabel("False Alarm Rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIG_E4, dpi=180)
        plt.close()

    print("[OUTPUT] 图已生成：")
    print(f"  {FIG_E1}")
    print(f"  {FIG_E2}")
    print(f"  {FIG_E3}")
    print(f"  {FIG_E4}")


# ============================================================
# 7. week2.md 报告初稿
# ============================================================

def write_week2_report(summary_df: pd.DataFrame, total_time_sec: float) -> None:
    """写第2周阶段总结初稿。"""
    WEEK2_MD.parent.mkdir(parents=True, exist_ok=True)

    report_cols = [
        "experiment_id",
        "condition_name",
        "num_scenes",
        "num_sources",
        "num_mics",
        "snr_db",
        "rt60_sec",
        "source_distance_cm",
        "mae_m",
        "p90_error_m",
        "error_std_m",
        "peak_ratio_mean",
        "peak_ratio_std",
        "hit_rate_10cm",
        "dual_hit_rate",
        "miss_rate",
        "false_alarm_rate",
        "runtime_mean_s",
        "is_control",
    ]

    lines = []

    lines.append("# 第2周定位方法与实验总结\n")

    lines.append("## 1. 本周目标\n")
    lines.append("第2周围绕鸡舍多麦克风声源定位开展实验，从 GCC-PHAT 与理论 TDOA 验证逐步推进到二维近场 SRP-PHAT、单源批处理定位和双源多峰定位。")
    lines.append("周五的重点是冻结默认参数，开展定位消融实验，比较麦克风数量、布局、SNR、源间距和 RT60 对定位性能的影响。\n")

    lines.append("## 2. 方法概述\n")
    lines.append("### 2.1 单源定位方法\n")
    lines.append("单源实验采用二维近场 SRP-PHAT 网格搜索方法。对每个候选网格点计算其到各麦克风的理论 TDOA，并在 GCC-PHAT 相关函数上累加对应延迟处的响应，得分最高的网格点作为预测声源位置。\n")

    lines.append("### 2.2 双源定位方法\n")
    lines.append("双源实验采用自动事件检测短时帧聚合 SRP-PHAT 方法。首先根据多通道短时能量自动检测主要声学事件，在事件附近选取高能短时帧；随后对每个短时帧执行单源 SRP-PHAT 定位，得到帧级定位点；再构建空间密度图和聚类先验图，融合为 hybrid_map；最后在 hybrid_map 上进行局部极大值检测和 NMS，输出两个预测峰，并用匈牙利算法匹配真实双源。\n")

    lines.append("## 3. Day1：GCC-PHAT 与理论 TDOA 验证\n")
    lines.append("Day1 完成 GCC-PHAT 与理论 TDOA 对齐验证，主要麦对误差满足不超过 1 sample 的验收要求。\n")

    lines.append("## 4. Day2：二维近场 SRP-PHAT 单点验证\n")
    lines.append("Day2 在无噪声、无混响、单源条件下完成二维近场 SRP-PHAT 功能验证，预测峰值与真实声源基本重合。\n")

    lines.append("## 5. Day3：单源批处理定位\n")
    lines.append("Day3 将单点 SRP-PHAT 扩展为多场景批处理，为 Day5 的 E1、E2、E3 和 E5 单源消融实验提供基础。\n")

    lines.append("## 6. Day4：双源多峰定位\n")
    lines.append("Day4 完成自动事件检测双源多峰定位流程，包括短时事件检测、逐帧 SRP-PHAT、density_map、cluster_prior_map、hybrid_map、局部极大值、NMS 和匈牙利匹配。\n")

    lines.append("## 7. Day5：定位消融实验\n")
    lines.append("Day5 进行五类消融实验：")
    lines.append("1. E1：麦克风数量消融，比较 4/6/8/12 麦。")
    lines.append("2. E2：布局消融，比较四角、两侧、单边和集中布局。")
    lines.append("3. E3：SNR 消融，比较 0/5/10/20/30 dB。")
    lines.append("4. E4：双源源间距消融，比较 10/20/30/50/80 cm。")
    lines.append("5. E5：RT60 消融，比较 0.1/0.3/0.5/0.7 s，统计误差和峰值比。\n")

    lines.append("## 8. 消融实验结果\n")
    lines.append(dataframe_to_markdown(summary_df[report_cols]))
    lines.append("")

    lines.append("## 9. 输出文件\n")
    lines.append(f"- 汇总结果：`{SUMMARY_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 逐场景结果：`{SCENE_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 配置文件：`{CONFIG_YAML.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 图1：`{FIG_E1.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 图2：`{FIG_E2.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 图3：`{FIG_E3.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 图4：`{FIG_E4.relative_to(PROJECT_ROOT)}`\n")

    lines.append("## 10. 阶段结论\n")
    lines.append("本阶段已经完成从 TDOA 验证到单源 SRP-PHAT、再到双源自动事件检测多峰定位的完整定位链路。Day5 消融实验用于分析不同硬件配置、噪声、混响和双源间距条件下的定位性能变化，为后续位置引导重建和系统评价提供依据。\n")

    lines.append("## 11. 下一步计划\n")
    lines.append("后续工作将基于定位结果开展位置引导的声学重建实验，并进一步比较 DAS、MVDR、LCMV 等波束形成方法在不同定位误差条件下的鲁棒性。\n")

    lines.append("## 12. 运行时间\n")
    lines.append(f"- Day5 总耗时：`{total_time_sec:.2f} s`\n")

    WEEK2_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OUTPUT] week2.md 已生成：{WEEK2_MD}")


# ============================================================
# 8. 配置与主流程
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第2周 Day5：定位消融与阶段总结。")

    parser.add_argument("--n-scenes", type=int, default=100, help="每个条件的场景数。测试可设为 3。")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--clean", action="store_true", help="运行前清空 results/week2/day5。")

    parser.add_argument(
        "--only",
        nargs="*",
        default=["E1", "E2", "E3", "E4", "E5"],
        choices=["E1", "E2", "E3", "E4", "E5"],
        help="只运行指定实验，例如 --only E1 E4。",
    )

    return parser.parse_args()


def write_config(args: argparse.Namespace) -> None:
    config = {
        "task": "week2_day5_localization_ablation",
        "script": "scripts/15_localization_ablation.py",
        "n_scenes_per_condition": int(args.n_scenes),
        "seed": int(args.seed),
        "default_params": {
            "layout": DEFAULT_LAYOUT,
            "snr_db": DEFAULT_SNR_DB,
            "rt60_sec": DEFAULT_RT60_SEC,
            "grid_spacing_m": DEFAULT_GRID_SPACING_M,
            "hit_radius_m": DEFAULT_HIT_RADIUS_M,
            "source_z": SOURCE_Z,
            "search_z": SEARCH_Z,
        },
        "experiments": {
            "E1_num_mics": [name for name, _ in E1_MIC_LAYOUTS],
            "E2_layout": [name for name, _ in E2_LAYOUTS],
            "E3_snr_db": E3_SNRS_DB,
            "E4_dual_distance_cm": E4_DISTANCES_CM,
            "E5_rt60_sec": E5_RT60S_SEC,
        },
        "outputs": {
            "scene_csv": str(SCENE_CSV.relative_to(PROJECT_ROOT)),
            "summary_csv": str(SUMMARY_CSV.relative_to(PROJECT_ROOT)),
            "week2_md": str(WEEK2_MD.relative_to(PROJECT_ROOT)),
            "figures": [
                str(FIG_E1.relative_to(PROJECT_ROOT)),
                str(FIG_E2.relative_to(PROJECT_ROOT)),
                str(FIG_E3.relative_to(PROJECT_ROOT)),
                str(FIG_E4.relative_to(PROJECT_ROOT)),
            ],
        },
    }

    save_yaml(config, CONFIG_YAML)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.clean:
        reset_output_dir(OUTPUT_DIR)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] 第2周 Day5：定位消融与阶段总结")
    print("=" * 80)
    print(f"[INFO] 项目根目录：{PROJECT_ROOT}")
    print(f"[INFO] Day3 脚本：{DAY3_SCRIPT}")
    print(f"[INFO] Day4 脚本：{DAY4_SCRIPT}")
    print(f"[INFO] 输出目录：{OUTPUT_DIR}")
    print(f"[INFO] 每组场景数：{args.n_scenes}")
    print(f"[INFO] 运行实验：{args.only}")

    check_layouts_exist()
    write_config(args)

    all_scene_dfs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    seed = int(args.seed)

    # ------------------------------------------------------------
    # E1：麦克风数量消融
    # ------------------------------------------------------------
    if "E1" in args.only:
        for idx, (layout_name, num_mics) in enumerate(E1_MIC_LAYOUTS):
            condition_name = layout_name

            scene_df, summary = run_single_condition(
                experiment_id="E1_num_mics",
                variable_name="num_mics",
                condition_name=condition_name,
                layout_name=layout_name,
                n_scenes=args.n_scenes,
                seed_base=seed + 100000 * (idx + 1),
                snr_db=DEFAULT_SNR_DB,
                rt60_sec=DEFAULT_RT60_SEC,
                source_z=SOURCE_Z,
                search_z=SEARCH_Z,
                grid_spacing_m=DEFAULT_GRID_SPACING_M,
                hit_radius_m=DEFAULT_HIT_RADIUS_M,
            )

            all_scene_dfs.append(scene_df)
            summary_rows.append(summary)

    # ------------------------------------------------------------
    # E2：布局消融
    # ------------------------------------------------------------
    if "E2" in args.only:
        for idx, (layout_name, layout_desc) in enumerate(E2_LAYOUTS):
            condition_name = layout_name

            scene_df, summary = run_single_condition(
                experiment_id="E2_layout",
                variable_name="layout",
                condition_name=condition_name,
                layout_name=layout_name,
                n_scenes=args.n_scenes,
                seed_base=seed + 200000 * (idx + 1),
                snr_db=DEFAULT_SNR_DB,
                rt60_sec=DEFAULT_RT60_SEC,
                source_z=SOURCE_Z,
                search_z=SEARCH_Z,
                grid_spacing_m=DEFAULT_GRID_SPACING_M,
                hit_radius_m=DEFAULT_HIT_RADIUS_M,
            )

            all_scene_dfs.append(scene_df)
            summary_rows.append(summary)

    # ------------------------------------------------------------
    # E3：SNR 消融
    # ------------------------------------------------------------
    if "E3" in args.only:
        for idx, snr_db in enumerate(E3_SNRS_DB):
            condition_name = f"snr_{int(snr_db)}db"

            scene_df, summary = run_single_condition(
                experiment_id="E3_snr",
                variable_name="snr_db",
                condition_name=condition_name,
                layout_name=DEFAULT_LAYOUT,
                n_scenes=args.n_scenes,
                seed_base=seed + 300000 * (idx + 1),
                snr_db=snr_db,
                rt60_sec=DEFAULT_RT60_SEC,
                source_z=SOURCE_Z,
                search_z=SEARCH_Z,
                grid_spacing_m=DEFAULT_GRID_SPACING_M,
                hit_radius_m=DEFAULT_HIT_RADIUS_M,
            )

            all_scene_dfs.append(scene_df)
            summary_rows.append(summary)

    # ------------------------------------------------------------
    # E4：双源源间距消融
    # ------------------------------------------------------------
    if "E4" in args.only:
        for idx, dist_cm in enumerate(E4_DISTANCES_CM):
            condition_name = f"dist_{int(dist_cm)}cm"

            scene_df, summary = run_dual_condition(
                experiment_id="E4_dual_distance",
                variable_name="source_distance_cm",
                condition_name=condition_name,
                target_distance_cm=dist_cm,
                n_scenes=args.n_scenes,
                seed_base=seed + 400000 * (idx + 1),
                layout_name=DEFAULT_LAYOUT,
                snr_db=DEFAULT_SNR_DB,
                rt60_sec=DEFAULT_RT60_SEC,
                source_z=SOURCE_Z,
                search_z=SEARCH_Z,
                grid_spacing_m=DEFAULT_GRID_SPACING_M,
                hit_radius_m=DEFAULT_HIT_RADIUS_M,
            )

            all_scene_dfs.append(scene_df)
            summary_rows.append(summary)

    # ------------------------------------------------------------
    # E5：RT60 消融
    # ------------------------------------------------------------
    if "E5" in args.only:
        for idx, rt60_sec in enumerate(E5_RT60S_SEC):
            condition_name = f"rt60_{rt60_sec:.2f}s"

            scene_df, summary = run_single_condition(
                experiment_id="E5_rt60",
                variable_name="rt60_sec",
                condition_name=condition_name,
                layout_name=DEFAULT_LAYOUT,
                n_scenes=args.n_scenes,
                seed_base=seed + 500000 * (idx + 1),
                snr_db=DEFAULT_SNR_DB,
                rt60_sec=rt60_sec,
                source_z=SOURCE_Z,
                search_z=SEARCH_Z,
                grid_spacing_m=DEFAULT_GRID_SPACING_M,
                hit_radius_m=DEFAULT_HIT_RADIUS_M,
            )

            all_scene_dfs.append(scene_df)
            summary_rows.append(summary)

    if len(summary_rows) == 0:
        raise RuntimeError("没有运行任何实验，请检查 --only 参数。")

    scene_all = pd.concat(all_scene_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_control_flag(summary_df)

    scene_all.to_csv(SCENE_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    generate_figures(summary_df)

    elapsed = time.time() - start_time
    write_week2_report(summary_df, total_time_sec=elapsed)

    print("\n" + "=" * 80)
    print("[RESULT] Day5 定位消融完成")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n[OUTPUT]")
    print(f"逐场景结果：{SCENE_CSV}")
    print(f"汇总结果：{SUMMARY_CSV}")
    print(f"配置文件：{CONFIG_YAML}")
    print(f"week2.md：{WEEK2_MD}")
    print(f"图1：{FIG_E1}")
    print(f"图2：{FIG_E2}")
    print(f"图3：{FIG_E3}")
    print(f"图4：{FIG_E4}")
    print(f"[TIME] 总耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
    main()