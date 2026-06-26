# -*- coding: utf-8 -*-
"""
第 2 周周二：二维近场 SRP-PHAT 声源定位

任务：
1. 建立 z=0 平面二维网格；
2. 预计算各网格点到各麦克风对的 TDOA 索引；
3. 生成 SRP-PHAT 热力图；
4. 验证无噪声单源定位峰值与真值一致；
5. 验收标准：无噪声单源二维定位误差 <= 5 cm。

运行：
    python scripts/12_nearfield_srp_2d.py

输出目录：
    results/week2/day2/12_nearfield_srp_2d/
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# =========================
# 1. 全局配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAGE_YAML = PROJECT_ROOT / "configs" / "week1" / "day4" / "cage.yaml"
MIC_LAYOUTS_YAML = PROJECT_ROOT / "configs" / "week1" / "day4" / "mic_layouts.yaml"

OUTPUT_DIR = PROJECT_ROOT / "results" / "week2" / "day2" / "12_nearfield_srp_2d"

MIC_LAYOUT_NAME = "mic_8"

FS = 48_000
SPEED_OF_SOUND = 343.0

# 按第 2 周周二要求：建立 z=0 平面二维网格
SEARCH_PLANE_Z = 0.0

# 无噪声单源场景的真实声源位置
# 注意：这里 z 固定为 0，是为了和“z=0 平面网格”保持一致
SOURCE_POSITION = np.array([0.42, 0.315, SEARCH_PLANE_Z], dtype=float)

# 网格间距：2 cm
GRID_SPACING_M = 0.02

# 网格边界留一点距离，避免声源贴墙
GRID_MARGIN_M = 0.02

# GCC-PHAT 插值倍数
INTERP = 32

# 探测信号配置
PROBE_DURATION_SEC = 2.0
RANDOM_SEED = 20260626

# 验收阈值
ERROR_THRESHOLD_CM = 5.0


# =========================
# 2. 工具函数
# =========================

def ensure_dir(path: Path) -> None:
    """创建目录。"""
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

    支持几种常见格式：
    1. layouts:
         mic_8:
           microphones:
             - mic_id: M01
               position: [x, y, z]

    2. layouts:
         mic_8:
           positions:
             - [x, y, z]

    3. layouts:
         mic_8:
           M01: [x, y, z]
           M02: [x, y, z]
    """
    layouts = mic_cfg.get("layouts", {})
    if layout_name not in layouts:
        raise KeyError(f"mic_layouts.yaml 中找不到布局：{layout_name}")

    layout = layouts[layout_name]

    if isinstance(layout, dict):
        mic_obj = (
            layout.get("microphones")
            or layout.get("mics")
            or layout.get("positions")
            or layout.get("mic_positions")
            or layout.get("coordinates")
        )

        # 如果没有 microphones/positions 字段，则尝试把 layout 自身当成 {M01: [x,y,z]} 结构
        if mic_obj is None:
            mic_obj = layout

    else:
        mic_obj = layout

    rows = []

    if isinstance(mic_obj, dict):
        for idx, (key, value) in enumerate(mic_obj.items(), start=1):
            mic_id = str(key)

            if isinstance(value, dict):
                pos = value.get("position") or value.get("pos") or value.get("xyz")
                if pos is None:
                    x = value.get("x")
                    y = value.get("y")
                    z = value.get("z")
                    pos = [x, y, z]
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
                    x = item.get("x")
                    y = item.get("y")
                    z = item.get("z")
                    pos = [x, y, z]

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


def next_power_of_two(n: int) -> int:
    """返回不小于 n 的最小 2 的幂。"""
    return 1 << (n - 1).bit_length()


def make_probe_signal(fs: int, duration_sec: float, seed: int) -> np.ndarray:
    """
    生成确定性宽带探测信号。

    用白噪声作为宽带信号，便于 GCC-PHAT 估计时间差。
    加一个淡入淡出窗，避免信号边界突变。
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_sec))

    signal = rng.standard_normal(n)

    # 加窗，避免起止边界突变
    fade_len = int(0.02 * fs)
    fade_len = max(1, min(fade_len, n // 10))

    window = np.ones(n)
    fade = np.linspace(0.0, 1.0, fade_len)
    window[:fade_len] = fade
    window[-fade_len:] = fade[::-1]

    signal = signal * window

    # 归一化
    signal = signal / (np.max(np.abs(signal)) + 1e-12)
    return signal.astype(np.float64)


def fractional_delay(signal: np.ndarray, delay_samples: float, out_len: int) -> np.ndarray:
    """
    使用频域相移实现分数采样延迟。

    delay_samples > 0 表示信号向后延迟。
    """
    n_fft = next_power_of_two(out_len + len(signal) + 4096)

    spectrum = np.fft.rfft(signal, n=n_fft)
    k = np.arange(len(spectrum), dtype=np.float64)

    # 频域相移：延迟 d 个采样点，对应乘 exp(-j 2π k d / N)
    phase = np.exp(-2j * np.pi * k * delay_samples / n_fft)
    delayed = np.fft.irfft(spectrum * phase, n=n_fft)

    return delayed[:out_len].astype(np.float64)


def simulate_direct_path_signals(
    source_signal: np.ndarray,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    sound_speed: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    生成无混响、无噪声、单源直达声多通道信号。

    每个麦克风信号 = 源信号经过传播延迟 + 距离衰减。
    """
    distances = np.linalg.norm(mic_positions - source_position[None, :], axis=1)
    delays_sec = distances / sound_speed
    delays_samples = delays_sec * fs

    max_delay_samples = int(math.ceil(np.max(delays_samples)))
    out_len = len(source_signal) + max_delay_samples + 2048

    channels = []
    for dist, delay_samp in zip(distances, delays_samples):
        y = fractional_delay(source_signal, delay_samp, out_len)

        # 简单距离衰减，避免近远通道幅度完全一样
        y = y / max(dist, 1e-6)

        channels.append(y)

    signals = np.stack(channels, axis=0)

    # 整体归一化
    signals = signals / (np.max(np.abs(signals)) + 1e-12)

    delay_table = pd.DataFrame({
        "mic_index": np.arange(len(mic_positions)),
        "distance_m": distances,
        "delay_sec": delays_sec,
        "delay_ms": delays_sec * 1000.0,
        "delay_samples": delays_samples,
    })

    return signals, delay_table


def gcc_phat_curve(
    sig_i: np.ndarray,
    sig_j: np.ndarray,
    fs: int,
    interp: int,
    max_tau_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算一对麦克风的 GCC-PHAT 曲线。

    返回：
        lags_samples: 横轴，单位为 samples
        cc: 纵轴，GCC-PHAT 相关响应

    符号约定：
        峰值 lag ≈ delay_i - delay_j
    """
    n = sig_i.size + sig_j.size

    sig_i_fft = np.fft.rfft(sig_i, n=n)
    sig_j_fft = np.fft.rfft(sig_j, n=n)

    cross_power = sig_i_fft * np.conj(sig_j_fft)
    cross_power = cross_power / (np.abs(cross_power) + 1e-12)

    cc_full = np.fft.irfft(cross_power, n=interp * n)

    max_shift = int(interp * n / 2)
    cc_full = np.concatenate((cc_full[-max_shift:], cc_full[:max_shift + 1]))

    lags_samples = np.arange(-max_shift, max_shift + 1, dtype=np.float64) / interp

    # 只保留物理上可能出现的 TDOA 范围，减少后续计算量
    keep = np.abs(lags_samples) <= max_tau_samples
    lags_samples = lags_samples[keep]
    cc = cc_full[keep]

    return lags_samples, cc


def build_mic_pairs(n_mics: int) -> List[Tuple[int, int]]:
    """生成所有麦克风对。"""
    pairs = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            pairs.append((i, j))
    return pairs


def make_grid(
    length_x: float,
    width_y: float,
    spacing: float,
    margin: float,
    z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成二维搜索网格。"""
    x_values = np.arange(margin, length_x - margin + 1e-9, spacing)
    y_values = np.arange(margin, width_y - margin + 1e-9, spacing)

    xx, yy = np.meshgrid(x_values, y_values)

    grid_points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(xx.size, z, dtype=np.float64),
    ])

    return x_values, y_values, grid_points


def float_index_from_lag(
    lag_samples: np.ndarray,
    lag_min: float,
    lag_step: float,
) -> np.ndarray:
    """
    把 TDOA 的 samples 值转换为 GCC-PHAT 曲线数组的浮点下标。

    例如：
        lag_min = -160
        lag_step = 1 / 32
        lag = -65

    那么 index 表示 lag=-65 在曲线数组里的位置。
    """
    return (lag_samples - lag_min) / lag_step


def sample_curve_by_float_index(curve: np.ndarray, index_float: np.ndarray) -> np.ndarray:
    """
    用线性插值从 GCC-PHAT 曲线中取值。

    index_float 可以是小数，因为 TDOA 不一定刚好落在整数采样点上。
    """
    idx0 = np.floor(index_float).astype(int)
    idx1 = idx0 + 1
    w = index_float - idx0

    valid = (idx0 >= 0) & (idx1 < len(curve))

    values = np.zeros_like(index_float, dtype=np.float64)
    values[valid] = (1.0 - w[valid]) * curve[idx0[valid]] + w[valid] * curve[idx1[valid]]

    return values


def srp_phat_grid_search(
    grid_points: np.ndarray,
    mic_positions: np.ndarray,
    mic_ids: List[str],
    pairs: List[Tuple[int, int]],
    gcc_cache: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    fs: int,
    sound_speed: float,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    对全部网格点计算 SRP-PHAT 得分。

    score(p) = 所有麦克风对在候选点理论 TDOA 位置上的 GCC-PHAT 相关值之和。
    """
    n_points = grid_points.shape[0]
    srp_scores = np.zeros(n_points, dtype=np.float64)

    pair_rows = []
    tdoa_index_cache = {}

    for pair_id, (i, j) in enumerate(pairs):
        mic_i = mic_positions[i]
        mic_j = mic_positions[j]

        # 计算每个候选点到两个麦克风的距离
        di = np.linalg.norm(grid_points - mic_i[None, :], axis=1)
        dj = np.linalg.norm(grid_points - mic_j[None, :], axis=1)

        # 候选点理论 TDOA，单位：samples
        # 含义：如果声源在该候选点，那么 Mi-Mj 应该出现这个到达时间差
        tdoa_samples = ((di - dj) / sound_speed) * fs

        cache = gcc_cache[(i, j)]
        lags_samples = cache["lags_samples"]
        cc = cache["cc"]

        lag_min = float(lags_samples[0])
        lag_step = float(lags_samples[1] - lags_samples[0])

        # 把理论 TDOA 转成 GCC-PHAT 曲线下标
        index_float = float_index_from_lag(tdoa_samples, lag_min, lag_step)

        # 在对应下标处取 GCC-PHAT 响应，并累加到 SRP 得分
        pair_score = sample_curve_by_float_index(cc, index_float)
        srp_scores += pair_score

        tdoa_index_cache[f"{mic_ids[i]}-{mic_ids[j]}"] = index_float.reshape(-1)

        # 记录该麦克风对自己的 GCC-PHAT 峰值，便于检查
        peak_idx = int(np.argmax(cc))
        pair_rows.append({
            "pair": f"{mic_ids[i]}-{mic_ids[j]}",
            "peak_lag_samples": float(lags_samples[peak_idx]),
            "peak_value": float(cc[peak_idx]),
            "lag_min_samples": float(lags_samples[0]),
            "lag_max_samples": float(lags_samples[-1]),
        })

    pair_peak_df = pd.DataFrame(pair_rows)

    return srp_scores, pair_peak_df, tdoa_index_cache


def plot_srp_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    score_map: np.ndarray,
    source_position: np.ndarray,
    pred_position: np.ndarray,
    mic_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制 SRP-PHAT 热力图。"""
    plt.figure(figsize=(8, 5))

    extent = [
        float(x_values[0]),
        float(x_values[-1]),
        float(y_values[0]),
        float(y_values[-1]),
    ]

    plt.imshow(
        score_map,
        origin="lower",
        extent=extent,
        aspect="auto",
    )

    plt.colorbar(label="SRP-PHAT score")

    plt.scatter(
        source_position[0],
        source_position[1],
        marker="x",
        s=120,
        label="True source",
    )

    plt.scatter(
        pred_position[0],
        pred_position[1],
        marker="o",
        s=80,
        facecolors="none",
        label="Predicted peak",
    )

    plt.scatter(
        mic_df["x"],
        mic_df["y"],
        marker="^",
        s=60,
        label="Microphones",
    )

    for _, row in mic_df.iterrows():
        plt.text(row["x"], row["y"], row["mic_id"], fontsize=8)

    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title("Near-field 2D SRP-PHAT Heatmap")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    output_path: Path,
    cage_dims: Dict[str, float],
    mic_df: pd.DataFrame,
    source_position: np.ndarray,
    pred_position: np.ndarray,
    error_cm: float,
    n_pairs: int,
    grid_spacing_m: float,
    grid_shape: Tuple[int, int],
    pass_flag: bool,
) -> None:
    """写 Markdown 报告。"""
    result_text = "通过" if pass_flag else "未通过"

    lines = []
    lines.append("# 第 2 周周二：二维近场 SRP-PHAT 声源定位报告\n")
    lines.append("## 1. 当前任务\n")
    lines.append("- 建立 z=0 平面二维网格。")
    lines.append("- 预计算各网格点到各麦克风对的 TDOA 索引。")
    lines.append("- 生成 SRP-PHAT 热力图。")
    lines.append("- 验证无噪声单源场景下，SRP-PHAT 峰值是否接近真实声源位置。")
    lines.append("- 验收标准：无噪声单源二维定位误差 ≤ 5 cm。\n")

    lines.append("## 2. 基准场景\n")
    lines.append(f"- 鸡笼尺寸：`{cage_dims['length_x']} × {cage_dims['width_y']} × {cage_dims['height_z']} m`")
    lines.append(f"- 采样率：`{FS} Hz`")
    lines.append(f"- 声速：`{SPEED_OF_SOUND} m/s`")
    lines.append(f"- 麦克风布局：`{MIC_LAYOUT_NAME}`")
    lines.append(f"- 麦克风数量：`{len(mic_df)}`")
    lines.append(f"- 麦克风对数量：`{n_pairs}`")
    lines.append(f"- 搜索平面：`z = {SEARCH_PLANE_Z} m`")
    lines.append(f"- 网格间距：`{grid_spacing_m} m`")
    lines.append(f"- 网格大小：`{grid_shape[1]} × {grid_shape[0]}`")
    lines.append(f"- 声源真实位置：`[{source_position[0]:.3f}, {source_position[1]:.3f}, {source_position[2]:.3f}] m`")
    lines.append("- 场景类型：无混响、无噪声、单源、直达声\n")

    lines.append("## 3. 定位结果\n")
    lines.append(f"- SRP-PHAT 峰值位置：`[{pred_position[0]:.3f}, {pred_position[1]:.3f}, {pred_position[2]:.3f}] m`")
    lines.append(f"- 二维定位误差：`{error_cm:.3f} cm`")
    lines.append(f"- 验收阈值：`≤ {ERROR_THRESHOLD_CM:.1f} cm`")
    lines.append(f"- 验收结论：`{result_text}`\n")

    lines.append("## 4. 方法说明\n")
    lines.append("对每个候选网格点 p，先计算该点到每一对麦克风的理论 TDOA：")
    lines.append("")
    lines.append("```text")
    lines.append("tau_ij(p) = [distance(p, Mi) - distance(p, Mj)] / speed_of_sound")
    lines.append("```")
    lines.append("")
    lines.append("然后把 tau_ij(p) 转成 GCC-PHAT 曲线下标，在对应位置读取相关值并累加：")
    lines.append("")
    lines.append("```text")
    lines.append("score(p) = sum_ij GCC_PHAT_ij(tau_ij(p))")
    lines.append("```")
    lines.append("")
    lines.append("最后取总分最高的网格点作为估计声源位置。\n")

    lines.append("## 5. 输出文件\n")
    lines.append("- SRP 热力图：`results/week2/day2/12_nearfield_srp_2d/srp_heatmap.png`")
    lines.append("- 网格得分表：`results/week2/day2/12_nearfield_srp_2d/srp_grid_result.csv`")
    lines.append("- 峰值结果表：`results/week2/day2/12_nearfield_srp_2d/srp_peak_result.csv`")
    lines.append("- 麦克风对 GCC 峰值表：`results/week2/day2/12_nearfield_srp_2d/gcc_pair_peak_table.csv`")
    lines.append("- TDOA 索引缓存：`results/week2/day2/12_nearfield_srp_2d/tdoa_index_cache.npz`")
    lines.append("- 场景配置：`results/week2/day2/12_nearfield_srp_2d/nearfield_srp_scene.yaml`")
    lines.append("- 报告：`results/week2/day2/12_nearfield_srp_2d/nearfield_srp_2d_report.md`\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# 3. 主流程
# =========================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("[INFO] 读取鸡笼与麦克风布局配置...")
    cage_cfg = load_yaml(CAGE_YAML)
    mic_cfg = load_yaml(MIC_LAYOUTS_YAML)

    cage_dims = get_cage_dimensions(cage_cfg)
    mic_df = extract_mic_layout(mic_cfg, MIC_LAYOUT_NAME)

    mic_positions = mic_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    mic_ids = mic_df["mic_id"].tolist()

    print(f"[INFO] 鸡笼尺寸：{cage_dims}")
    print(f"[INFO] 麦克风数量：{len(mic_df)}")
    print(f"[INFO] 搜索平面 z = {SEARCH_PLANE_Z}")

    # 检查声源是否在鸡笼范围内
    sx, sy, sz = SOURCE_POSITION
    if not (0.0 <= sx <= cage_dims["length_x"] and 0.0 <= sy <= cage_dims["width_y"]):
        raise ValueError(f"声源二维坐标超出鸡笼范围：{SOURCE_POSITION}")

    print("[INFO] 生成无噪声单源直达声信号...")
    probe = make_probe_signal(FS, PROBE_DURATION_SEC, RANDOM_SEED)

    multichannel, delay_table = simulate_direct_path_signals(
        source_signal=probe,
        source_position=SOURCE_POSITION,
        mic_positions=mic_positions,
        fs=FS,
        sound_speed=SPEED_OF_SOUND,
    )

    delay_table.insert(0, "mic_id", mic_ids)
    delay_table.to_csv(OUTPUT_DIR / "mic_delay_table.csv", index=False, encoding="utf-8-sig")

    print("[INFO] 生成所有麦克风对...")
    pairs = build_mic_pairs(len(mic_df))
    print(f"[INFO] 麦克风对数量：{len(pairs)}")

    # 根据阵列最大距离估计物理上可能的最大 TDOA
    max_mic_dist = 0.0
    for i, j in pairs:
        dist = float(np.linalg.norm(mic_positions[i] - mic_positions[j]))
        max_mic_dist = max(max_mic_dist, dist)

    max_tau_samples = int(math.ceil((max_mic_dist / SPEED_OF_SOUND) * FS)) + 10
    print(f"[INFO] 最大物理 TDOA 范围：±{max_tau_samples} samples")

    print("[INFO] 计算每个麦克风对的 GCC-PHAT 曲线...")
    gcc_cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    for i, j in pairs:
        lags_samples, cc = gcc_phat_curve(
            sig_i=multichannel[i],
            sig_j=multichannel[j],
            fs=FS,
            interp=INTERP,
            max_tau_samples=max_tau_samples,
        )

        gcc_cache[(i, j)] = {
            "lags_samples": lags_samples,
            "cc": cc,
        }

    print("[INFO] 建立 z=0 平面二维搜索网格...")
    x_values, y_values, grid_points = make_grid(
        length_x=cage_dims["length_x"],
        width_y=cage_dims["width_y"],
        spacing=GRID_SPACING_M,
        margin=GRID_MARGIN_M,
        z=SEARCH_PLANE_Z,
    )

    print(f"[INFO] 网格点数量：{len(grid_points)}")

    print("[INFO] 预计算 TDOA 索引并计算 SRP-PHAT 得分...")
    srp_scores, pair_peak_df, tdoa_index_cache = srp_phat_grid_search(
        grid_points=grid_points,
        mic_positions=mic_positions,
        mic_ids=mic_ids,
        pairs=pairs,
        gcc_cache=gcc_cache,
        fs=FS,
        sound_speed=SPEED_OF_SOUND,
    )

    # 保存 TDOA 索引缓存
    np.savez_compressed(OUTPUT_DIR / "tdoa_index_cache.npz", **tdoa_index_cache)

    # 找到 SRP-PHAT 最大峰值
    best_idx = int(np.argmax(srp_scores))
    pred_position = grid_points[best_idx]

    error_cm = float(np.linalg.norm(pred_position[:2] - SOURCE_POSITION[:2]) * 100.0)
    pass_flag = error_cm <= ERROR_THRESHOLD_CM

    print("[INFO] 保存网格得分结果...")
    grid_result = pd.DataFrame({
        "x": grid_points[:, 0],
        "y": grid_points[:, 1],
        "z": grid_points[:, 2],
        "srp_score": srp_scores,
    })
    grid_result.to_csv(OUTPUT_DIR / "srp_grid_result.csv", index=False, encoding="utf-8-sig")

    peak_result = pd.DataFrame([{
        "true_x": SOURCE_POSITION[0],
        "true_y": SOURCE_POSITION[1],
        "true_z": SOURCE_POSITION[2],
        "pred_x": pred_position[0],
        "pred_y": pred_position[1],
        "pred_z": pred_position[2],
        "error_cm": error_cm,
        "threshold_cm": ERROR_THRESHOLD_CM,
        "pass": pass_flag,
        "grid_spacing_m": GRID_SPACING_M,
        "num_mics": len(mic_df),
        "num_pairs": len(pairs),
    }])
    peak_result.to_csv(OUTPUT_DIR / "srp_peak_result.csv", index=False, encoding="utf-8-sig")

    pair_peak_df.to_csv(OUTPUT_DIR / "gcc_pair_peak_table.csv", index=False, encoding="utf-8-sig")

    print("[INFO] 绘制 SRP-PHAT 热力图...")
    score_map = srp_scores.reshape(len(y_values), len(x_values))

    plot_srp_heatmap(
        x_values=x_values,
        y_values=y_values,
        score_map=score_map,
        source_position=SOURCE_POSITION,
        pred_position=pred_position,
        mic_df=mic_df,
        output_path=OUTPUT_DIR / "srp_heatmap.png",
    )

    scene_yaml = {
        "task": "week2_day2_nearfield_srp_phat_2d",
        "script": "scripts/12_nearfield_srp_2d.py",
        "scene_type": "anechoic_single_source_direct_path",
        "fs": FS,
        "speed_of_sound": SPEED_OF_SOUND,
        "cage_yaml": str(CAGE_YAML.relative_to(PROJECT_ROOT)),
        "mic_layouts_yaml": str(MIC_LAYOUTS_YAML.relative_to(PROJECT_ROOT)),
        "mic_layout": MIC_LAYOUT_NAME,
        "source_position": SOURCE_POSITION.tolist(),
        "search_grid": {
            "plane_z": SEARCH_PLANE_Z,
            "spacing_m": GRID_SPACING_M,
            "margin_m": GRID_MARGIN_M,
            "x_min": float(x_values[0]),
            "x_max": float(x_values[-1]),
            "y_min": float(y_values[0]),
            "y_max": float(y_values[-1]),
            "num_x": int(len(x_values)),
            "num_y": int(len(y_values)),
            "num_points": int(len(grid_points)),
        },
        "gcc_phat": {
            "interp": INTERP,
            "num_pairs": len(pairs),
            "max_tau_samples": max_tau_samples,
        },
        "result": {
            "pred_position": pred_position.tolist(),
            "error_cm": error_cm,
            "threshold_cm": ERROR_THRESHOLD_CM,
            "pass": bool(pass_flag),
        },
        "outputs": {
            "srp_heatmap": str((OUTPUT_DIR / "srp_heatmap.png").relative_to(PROJECT_ROOT)),
            "srp_grid_result": str((OUTPUT_DIR / "srp_grid_result.csv").relative_to(PROJECT_ROOT)),
            "srp_peak_result": str((OUTPUT_DIR / "srp_peak_result.csv").relative_to(PROJECT_ROOT)),
            "gcc_pair_peak_table": str((OUTPUT_DIR / "gcc_pair_peak_table.csv").relative_to(PROJECT_ROOT)),
            "tdoa_index_cache": str((OUTPUT_DIR / "tdoa_index_cache.npz").relative_to(PROJECT_ROOT)),
            "report": str((OUTPUT_DIR / "nearfield_srp_2d_report.md").relative_to(PROJECT_ROOT)),
        },
        "note": "本脚本完成第2周周二二维近场SRP-PHAT验证，成果放在 results/week2/day2/12_nearfield_srp_2d。",
    }
    save_yaml(scene_yaml, OUTPUT_DIR / "nearfield_srp_scene.yaml")

    write_report(
        output_path=OUTPUT_DIR / "nearfield_srp_2d_report.md",
        cage_dims=cage_dims,
        mic_df=mic_df,
        source_position=SOURCE_POSITION,
        pred_position=pred_position,
        error_cm=error_cm,
        n_pairs=len(pairs),
        grid_spacing_m=GRID_SPACING_M,
        grid_shape=score_map.shape,
        pass_flag=pass_flag,
    )

    print("\n========== 第 2 周周二验收结果 ==========")
    print(f"真实位置：[{SOURCE_POSITION[0]:.3f}, {SOURCE_POSITION[1]:.3f}, {SOURCE_POSITION[2]:.3f}] m")
    print(f"预测位置：[{pred_position[0]:.3f}, {pred_position[1]:.3f}, {pred_position[2]:.3f}] m")
    print(f"二维误差：{error_cm:.3f} cm")
    print(f"验收阈值：<= {ERROR_THRESHOLD_CM:.1f} cm")
    print(f"是否通过：{pass_flag}")
    print(f"输出目录：{OUTPUT_DIR}")

    if pass_flag:
        print("[PASS] 无噪声单源 SRP-PHAT 二维定位误差满足 <= 5 cm。")
    else:
        print("[FAIL] 无噪声单源 SRP-PHAT 二维定位误差超过 5 cm，需要检查符号约定、网格高度或麦克风布局。")


if __name__ == "__main__":
    main()