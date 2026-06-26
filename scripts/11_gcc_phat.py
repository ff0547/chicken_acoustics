# -*- coding: utf-8 -*-
"""
第 2 周周一：GCC-PHAT 与理论 TDOA

脚本名称：
scripts/day1.py

任务要求：
1. 实现麦克风对 GCC-PHAT；
2. 在无混响单源场景比较理论 TDOA 和估计 TDOA；
3. 检查采样点量化误差；
4. 输出 TDOA 误差表和曲线；
5. 验收标准：主要麦对 TDOA 误差 ≤ 1 个采样点，若超过必须给出解释。

注意：
- 本脚本是第 2 周周一任务；
- 本脚本不生成论文级 E1～E10 大规模场景；
- 本脚本不依赖 11_build_paper_sim_scenes.py；
- 当前只验证 GCC-PHAT 的 TDOA 估计是否正确；
- 后续单源/双源二维定位会在后面的脚本中完成。
"""

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 路径配置
# =========================

ROOT = Path(__file__).resolve().parents[1]

CAGE_YAML = ROOT / "configs" / "week1" / "day4" / "cage.yaml"
MIC_LAYOUTS_YAML = ROOT / "configs" / "week1" / "day4" / "mic_layouts.yaml"

RESULT_DIR = ROOT / "results" / "week2" / "day1"

TDOA_TABLE_CSV = RESULT_DIR / "tdoa_error_table.csv"
TDOA_CURVE_PNG = RESULT_DIR / "tdoa_error_curve.png"
GCC_MAIN_PAIR_PNG = RESULT_DIR / "gcc_phat_main_pair_curve.png"
SCENE_CONFIG_YAML = RESULT_DIR / "anechoic_tdoa_scene.yaml"
REPORT_MD = RESULT_DIR / "gcc_phat_report.md"


# =========================
# 论文级基准参数
# =========================

EXPECTED_CAGE_SIZE = {
    "length_x": 1.20,
    "width_y": 0.75,
    "height_z": 0.60,
}

FS = 48000
SPEED_OF_SOUND = 343.0

# 基准麦克风布局：8 麦，鸡笼四周分布式
BASELINE_MIC_LAYOUT = "mic_8"

# 发声平面：论文级基准 z = 0.35 m
SOURCE_Z_BASELINE = 0.35

# 固定声源位置比例，避免刚好位于中心导致部分麦对 TDOA 太小
SOURCE_X_RATIO = 0.35
SOURCE_Y_RATIO = 0.42

# 宽带探测信号，用于 GCC-PHAT 验证
PROBE_SIGNAL_DURATION_SEC = 2.0
PROBE_RANDOM_SEED = 20260626

# 额外尾部时长，避免延迟后的信号被截断
TAIL_SEC = 0.20

# GCC-PHAT 插值倍数
GCC_INTERP = 32

# 主麦对选择方式：理论 TDOA 绝对值最大的麦克风对
MAIN_PAIR_RULE = "max_abs_theoretical_tdoa"

# 验收阈值：主要麦对误差不超过 1 个采样点
MAIN_PAIR_ERROR_THRESHOLD_SAMPLES = 1.0


# =========================
# 基础工具函数
# =========================

def rel_to_root(path: Path) -> str:
    """返回相对于项目根目录的路径字符串。"""
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def load_yaml(path: Path) -> Dict:
    """读取 YAML 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Path):
    """保存 YAML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def ensure_dirs():
    """创建输出目录。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def check_baseline_cage(cage: Dict):
    """
    检查第 4 天 cage.yaml 是否已经更新为论文级基准尺寸。

    如果这里报错，说明需要先重新运行：
    python scripts\\04_cage_mic_layouts.py
    """
    dims = cage["dimensions"]

    for key, expected_value in EXPECTED_CAGE_SIZE.items():
        actual_value = float(dims[key])
        if abs(actual_value - expected_value) > 1e-6:
            raise ValueError(
                f"cage.yaml 中 {key}={actual_value}，"
                f"但论文级基准要求为 {expected_value}。"
                f"请先更新并运行 scripts\\04_cage_mic_layouts.py。"
            )

    height_z = float(dims["height_z"])
    if SOURCE_Z_BASELINE > height_z:
        raise ValueError(
            f"发声平面 z={SOURCE_Z_BASELINE} 超出鸡笼高度 {height_z}。"
        )


def get_mic_positions(mic_layouts: Dict, layout_name: str) -> Tuple[List[str], np.ndarray]:
    """
    读取指定麦克风布局。

    返回：
    - mic_ids：麦克风 ID 列表
    - mic_positions：[num_mics, 3] 坐标矩阵
    """
    if layout_name not in mic_layouts["layouts"]:
        raise KeyError(
            f"mic_layouts.yaml 中缺少 {layout_name}。"
            f"请先确认第4天麦克风布局脚本是否已重新运行。"
        )

    microphones = mic_layouts["layouts"][layout_name]["microphones"]

    mic_ids = [m["id"] for m in microphones]
    mic_positions = np.array([m["position"] for m in microphones], dtype=np.float64)

    return mic_ids, mic_positions


def build_source_position(cage: Dict) -> np.ndarray:
    """
    构造无混响单源验证场景中的真实声源位置。

    使用固定比例位置，保证可复现。
    """
    length_x = float(cage["dimensions"]["length_x"])
    width_y = float(cage["dimensions"]["width_y"])
    height_z = float(cage["dimensions"]["height_z"])

    x = length_x * SOURCE_X_RATIO
    y = width_y * SOURCE_Y_RATIO
    z = SOURCE_Z_BASELINE

    if not (0 <= x <= length_x and 0 <= y <= width_y and 0 <= z <= height_z):
        raise ValueError(f"声源位置非法: {[x, y, z]}")

    return np.array([x, y, z], dtype=np.float64)


def generate_probe_signal(fs: int = FS) -> np.ndarray:
    """
    生成宽带探测信号。

    当前任务验证 TDOA 估计，不验证鸡叫分类。
    使用固定随机种子的宽带信号可以得到更清晰的 GCC-PHAT 峰值。
    """
    rng = np.random.default_rng(PROBE_RANDOM_SEED)

    num_samples = int(PROBE_SIGNAL_DURATION_SEC * fs)

    y = rng.normal(0.0, 1.0, size=num_samples).astype(np.float64)

    # 加 Hann 窗，减少起止突变
    window = np.hanning(num_samples)
    y = y * window

    # 去直流
    y = y - np.mean(y)

    # 峰值归一化
    peak = np.max(np.abs(y))
    if peak > 1e-12:
        y = y / peak * 0.8

    return y.astype(np.float32)


def fractional_delay_linear(
    signal: np.ndarray,
    delay_samples: float,
    out_len: int
) -> np.ndarray:
    """
    使用线性插值实现分数采样延迟。

    delay_samples > 0 表示信号向后延迟。
    """
    source_index = np.arange(len(signal), dtype=np.float64)
    target_index = np.arange(out_len, dtype=np.float64) - float(delay_samples)

    delayed = np.interp(
        target_index,
        source_index,
        signal.astype(np.float64),
        left=0.0,
        right=0.0,
    )

    return delayed.astype(np.float32)


def build_anechoic_multichannel_scene(
    source_signal: np.ndarray,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    fs: int = FS,
    sound_speed: float = SPEED_OF_SOUND
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造无混响单源多通道信号。

    只保留直达声：
    mic_signal_m(t) = source(t - distance_m / c) / distance_m

    返回：
    - multichannel：[num_mics, num_samples]
    - distances：[num_mics]
    - delays_sec：[num_mics]
    """
    distances = np.linalg.norm(mic_positions - source_position[None, :], axis=1)
    delays_sec = distances / sound_speed
    delays_samples = delays_sec * fs

    max_delay_samples = int(np.ceil(np.max(delays_samples)))
    tail_samples = int(TAIL_SEC * fs)
    out_len = len(source_signal) + max_delay_samples + tail_samples

    num_mics = mic_positions.shape[0]
    multichannel = np.zeros((num_mics, out_len), dtype=np.float32)

    for m in range(num_mics):
        delayed = fractional_delay_linear(
            signal=source_signal,
            delay_samples=float(delays_samples[m]),
            out_len=out_len,
        )

        # 距离衰减
        attenuation = 1.0 / max(float(distances[m]), 0.05)
        multichannel[m] = delayed * attenuation

    # 全局归一化，避免幅度过大
    peak = np.max(np.abs(multichannel))
    if peak > 1e-12:
        multichannel = (multichannel / peak * 0.9).astype(np.float32)

    return multichannel, distances, delays_sec


# =========================
# GCC-PHAT 核心
# =========================

def parabolic_peak_offset(y: np.ndarray, idx: int) -> float:
    """
    对相关峰做抛物线插值，返回亚采样偏移。
    """
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0

    left = float(y[idx - 1])
    center = float(y[idx])
    right = float(y[idx + 1])

    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0

    offset = 0.5 * (left - right) / denom

    if not np.isfinite(offset):
        return 0.0

    return float(np.clip(offset, -1.0, 1.0))


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int = FS,
    max_tau: float = None,
    interp: int = GCC_INTERP
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    GCC-PHAT 估计两个信号之间的 TDOA。

    参数：
    - sig：待估计通道信号
    - refsig：参考通道信号
    - fs：采样率
    - max_tau：最大允许时延，单位秒
    - interp：插值倍数

    返回：
    - tau：估计时延，单位秒
    - lags_sec：相关函数横轴，单位秒
    - cc_abs：GCC-PHAT 相关函数绝对值
    - peak_value：峰值大小

    约定：
    - tau > 0 表示 sig 相对 refsig 更晚到达；
    - 理论值使用 delay_i - delay_j。
    """
    sig = np.asarray(sig, dtype=np.float64)
    refsig = np.asarray(refsig, dtype=np.float64)

    n = sig.size + refsig.size

    nfft = 1
    while nfft < n:
        nfft *= 2

    sig_fft = np.fft.rfft(sig, n=nfft)
    ref_fft = np.fft.rfft(refsig, n=nfft)

    cross_power = sig_fft * np.conj(ref_fft)
    cross_power_abs = np.abs(cross_power)

    cross_power = cross_power / (cross_power_abs + 1e-15)

    cc = np.fft.irfft(cross_power, n=interp * nfft)

    max_shift = int(interp * nfft / 2)

    if max_tau is not None:
        max_shift = min(max_shift, int(interp * fs * max_tau))

    # 拼接成 [-max_shift, +max_shift] 的搜索范围
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    cc_abs = np.abs(cc)
    peak_idx = int(np.argmax(cc_abs))

    shift = peak_idx - max_shift

    # 抛物线插值细化峰值
    sub_offset = parabolic_peak_offset(cc_abs, peak_idx)
    shift_refined = shift + sub_offset

    tau = shift_refined / float(interp * fs)

    lags_sec = np.arange(-max_shift, max_shift + 1, dtype=np.float64) / float(interp * fs)
    peak_value = float(cc_abs[peak_idx])

    return float(tau), lags_sec, cc_abs.astype(np.float64), peak_value


# =========================
# TDOA 理论值与误差计算
# =========================

def compute_theoretical_tdoa(
    delays_sec: np.ndarray,
    i: int,
    j: int
) -> float:
    """
    计算理论 TDOA。

    定义：
    TDOA(i, j) = delay_i - delay_j
    """
    return float(delays_sec[i] - delays_sec[j])


def build_pair_records(
    mic_ids: List[str],
    mic_positions: np.ndarray,
    multichannel: np.ndarray,
    delays_sec: np.ndarray,
    fs: int = FS
) -> Tuple[pd.DataFrame, Dict]:
    """
    对所有麦克风对计算理论 TDOA、GCC-PHAT 估计 TDOA 和误差。
    """
    num_mics = len(mic_ids)

    max_mic_distance = 0.0
    for a in range(num_mics):
        for b in range(num_mics):
            d = float(np.linalg.norm(mic_positions[a] - mic_positions[b]))
            max_mic_distance = max(max_mic_distance, d)

    max_tau = max_mic_distance / SPEED_OF_SOUND + 0.001

    raw_records = []
    gcc_cache = {}

    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            mic_i = mic_ids[i]
            mic_j = mic_ids[j]

            theory_tdoa_sec = compute_theoretical_tdoa(delays_sec, i, j)
            theory_tdoa_samples = theory_tdoa_sec * fs
            theory_tdoa_round_samples = round(theory_tdoa_samples)
            quantization_error_samples = theory_tdoa_round_samples - theory_tdoa_samples

            est_tdoa_sec_raw, lags_sec, cc_abs, peak_value = gcc_phat(
                sig=multichannel[i],
                refsig=multichannel[j],
                fs=fs,
                max_tau=max_tau,
                interp=GCC_INTERP,
            )

            gcc_cache[(mic_i, mic_j)] = {
                "lags_sec": lags_sec,
                "cc_abs": cc_abs,
                "peak_value": peak_value,
            }

            raw_records.append({
                "mic_i": mic_i,
                "mic_j": mic_j,
                "pair": f"{mic_i}-{mic_j}",
                "theory_tdoa_sec": theory_tdoa_sec,
                "theory_tdoa_samples": theory_tdoa_samples,
                "theory_tdoa_round_samples": theory_tdoa_round_samples,
                "quantization_error_samples": quantization_error_samples,
                "est_tdoa_sec_raw": est_tdoa_sec_raw,
                "est_tdoa_samples_raw": est_tdoa_sec_raw * fs,
                "gcc_peak_value": peak_value,
            })

    df_raw = pd.DataFrame(raw_records)

    # 自动统一符号约定
    raw_err = np.median(
        np.abs(df_raw["est_tdoa_samples_raw"] - df_raw["theory_tdoa_samples"])
    )
    neg_err = np.median(
        np.abs(-df_raw["est_tdoa_samples_raw"] - df_raw["theory_tdoa_samples"])
    )

    if neg_err < raw_err:
        sign_correction = -1.0
        sign_note = "GCC-PHAT 原始符号与理论定义相反，已乘以 -1 做统一。"
    else:
        sign_correction = 1.0
        sign_note = "GCC-PHAT 原始符号与理论定义一致。"

    df = df_raw.copy()
    df["sign_correction"] = sign_correction
    df["est_tdoa_sec"] = df["est_tdoa_sec_raw"] * sign_correction
    df["est_tdoa_samples"] = df["est_tdoa_samples_raw"] * sign_correction

    df["error_sec"] = df["est_tdoa_sec"] - df["theory_tdoa_sec"]
    df["error_samples"] = df["est_tdoa_samples"] - df["theory_tdoa_samples"]
    df["abs_error_samples"] = np.abs(df["error_samples"])
    df["abs_error_sec"] = np.abs(df["error_sec"])

    df["pass_le_1_sample"] = df["abs_error_samples"] <= MAIN_PAIR_ERROR_THRESHOLD_SAMPLES

    main_idx = int(np.argmax(np.abs(df["theory_tdoa_samples"].to_numpy())))
    df["is_main_pair"] = False
    df.loc[main_idx, "is_main_pair"] = True

    main_pair = df.loc[main_idx, "pair"]

    summary = {
        "max_tau_search_sec": float(max_tau),
        "sign_correction": float(sign_correction),
        "sign_note": sign_note,
        "main_pair": str(main_pair),
        "main_pair_abs_error_samples": float(df.loc[main_idx, "abs_error_samples"]),
        "main_pair_pass": bool(df.loc[main_idx, "pass_le_1_sample"]),
        "all_pairs_max_abs_error_samples": float(df["abs_error_samples"].max()),
        "all_pairs_mean_abs_error_samples": float(df["abs_error_samples"].mean()),
        "all_pairs_pass_le_1_sample": bool(df["pass_le_1_sample"].all()),
    }

    return df, {
        "summary": summary,
        "gcc_cache": gcc_cache,
    }


# =========================
# 绘图
# =========================

def plot_tdoa_error_curve(df: pd.DataFrame, out_path: Path):
    """绘制所有麦克风对的 TDOA 估计误差曲线。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values("abs_error_samples", ascending=False).reset_index(drop=True)

    x = np.arange(len(plot_df))
    y = plot_df["error_samples"].to_numpy()

    plt.figure(figsize=(14, 5))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.axhline(-1.0, linestyle="--", linewidth=1)
    plt.axhline(0.0, linewidth=1)

    plt.xticks(x, plot_df["pair"].tolist(), rotation=60, ha="right")
    plt.xlabel("Microphone pair")
    plt.ylabel("TDOA estimation error / samples")
    plt.title("GCC-PHAT TDOA Error for All Microphone Pairs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_main_pair_gcc_curve(
    df: pd.DataFrame,
    gcc_cache: Dict,
    out_path: Path
):
    """绘制主要麦对的 GCC-PHAT 相关峰。"""
    main_row = df[df["is_main_pair"]].iloc[0]
    mic_i = main_row["mic_i"]
    mic_j = main_row["mic_j"]

    data = gcc_cache[(mic_i, mic_j)]
    lags_ms = data["lags_sec"] * 1000.0
    cc_abs = data["cc_abs"]

    theory_ms = main_row["theory_tdoa_sec"] * 1000.0
    est_ms = main_row["est_tdoa_sec"] * 1000.0

    plt.figure(figsize=(10, 5))
    plt.plot(lags_ms, cc_abs, linewidth=1)
    plt.axvline(theory_ms, linestyle="--", linewidth=1, label="theoretical TDOA")
    plt.axvline(est_ms, linestyle=":", linewidth=1, label="estimated TDOA")
    plt.legend()

    plt.xlabel("Lag / ms")
    plt.ylabel("Abs GCC-PHAT correlation")
    plt.title(f"GCC-PHAT Peak of Main Pair: {mic_i}-{mic_j}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# 报告
# =========================

def write_report(
    cage: Dict,
    mic_ids: List[str],
    mic_positions: np.ndarray,
    source_position: np.ndarray,
    distances: np.ndarray,
    delays_sec: np.ndarray,
    df: pd.DataFrame,
    summary: Dict
):
    """生成 Markdown 报告。"""
    main_row = df[df["is_main_pair"]].iloc[0]

    lines = []

    lines.append("# 第 2 周周一：GCC-PHAT 与理论 TDOA 验证报告")
    lines.append("")
    lines.append("## 1. 当前任务")
    lines.append("")
    lines.append("- 实现麦克风对 GCC-PHAT。")
    lines.append("- 在无混响单源场景中比较理论 TDOA 与估计 TDOA。")
    lines.append("- 检查采样点量化误差。")
    lines.append("- 验收标准：主要麦对 TDOA 误差 ≤ 1 个采样点，若超过需要解释。")
    lines.append("")
    lines.append("说明：本任务只验证 TDOA 估计，不生成论文级 E1～E10 场景矩阵。")
    lines.append("")

    lines.append("## 2. 基准场景")
    lines.append("")
    lines.append(
        f"- 鸡笼尺寸：`{cage['dimensions']['length_x']} × "
        f"{cage['dimensions']['width_y']} × {cage['dimensions']['height_z']} m`"
    )
    lines.append(f"- 采样率：`{FS} Hz`")
    lines.append(f"- 声速：`{SPEED_OF_SOUND} m/s`")
    lines.append(f"- 麦克风布局：`{BASELINE_MIC_LAYOUT}`")
    lines.append(f"- 麦克风数量：`{len(mic_ids)}`")
    lines.append(
        f"- 声源位置：`[{source_position[0]:.3f}, "
        f"{source_position[1]:.3f}, {source_position[2]:.3f}] m`"
    )
    lines.append("- 场景类型：无混响、单源、直达声")
    lines.append("")

    lines.append("## 3. 麦克风距离与到达时间")
    lines.append("")
    lines.append("| mic_id | x | y | z | distance_m | delay_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for idx, mic_id in enumerate(mic_ids):
        x, y, z = mic_positions[idx]
        lines.append(
            f"| {mic_id} | {x:.3f} | {y:.3f} | {z:.3f} | "
            f"{distances[idx]:.6f} | {delays_sec[idx] * 1000:.6f} |"
        )

    lines.append("")

    lines.append("## 4. GCC-PHAT 符号约定")
    lines.append("")
    lines.append("- 理论 TDOA 定义：`delay_i - delay_j`。")
    lines.append(f"- 符号校正：`{summary['sign_correction']}`")
    lines.append(f"- 说明：{summary['sign_note']}")
    lines.append("")

    lines.append("## 5. 主要麦对验收结果")
    lines.append("")
    lines.append(f"- 主要麦对选择规则：`{MAIN_PAIR_RULE}`")
    lines.append(f"- 主要麦对：`{summary['main_pair']}`")
    lines.append(f"- 理论 TDOA：`{main_row['theory_tdoa_samples']:.4f}` samples")
    lines.append(f"- GCC-PHAT 估计 TDOA：`{main_row['est_tdoa_samples']:.4f}` samples")
    lines.append(f"- 误差：`{main_row['error_samples']:.4f}` samples")
    lines.append(f"- 绝对误差：`{main_row['abs_error_samples']:.4f}` samples")
    lines.append(f"- 验收阈值：`≤ {MAIN_PAIR_ERROR_THRESHOLD_SAMPLES} sample`")
    lines.append("")

    if summary["main_pair_pass"]:
        lines.append("结论：主要麦对 TDOA 误差满足 ≤ 1 个采样点，今日验收通过。")
    else:
        lines.append("结论：主要麦对 TDOA 误差超过 1 个采样点，需要检查信号带宽、峰值位置或符号约定。")

    lines.append("")

    lines.append("## 6. 全部麦对统计")
    lines.append("")
    lines.append(f"- 全部麦对最大绝对误差：`{summary['all_pairs_max_abs_error_samples']:.4f}` samples")
    lines.append(f"- 全部麦对平均绝对误差：`{summary['all_pairs_mean_abs_error_samples']:.4f}` samples")
    lines.append(f"- 全部麦对是否均 ≤ 1 sample：`{summary['all_pairs_pass_le_1_sample']}`")
    lines.append("")

    lines.append("## 7. 输出文件")
    lines.append("")
    lines.append(f"- TDOA 误差表：`{rel_to_root(TDOA_TABLE_CSV)}`")
    lines.append(f"- TDOA 误差曲线：`{rel_to_root(TDOA_CURVE_PNG)}`")
    lines.append(f"- 主要麦对 GCC-PHAT 曲线：`{rel_to_root(GCC_MAIN_PAIR_PNG)}`")
    lines.append(f"- 场景配置：`{rel_to_root(SCENE_CONFIG_YAML)}`")
    lines.append(f"- 报告：`{rel_to_root(REPORT_MD)}`")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# =========================
# 主函数
# =========================

def main():
    """主函数。"""
    warnings.filterwarnings("ignore")
    ensure_dirs()

    print("========== 第 2 周周一：GCC-PHAT 与理论 TDOA ==========")
    print(f"[INFO] 项目根目录: {ROOT}")
    print(f"[INFO] cage 配置: {CAGE_YAML}")
    print(f"[INFO] mic layouts 配置: {MIC_LAYOUTS_YAML}")
    print(f"[INFO] 输出目录: {RESULT_DIR}")
    print("")

    if not CAGE_YAML.exists():
        raise FileNotFoundError(
            f"缺少 {CAGE_YAML}，请先运行 scripts\\04_cage_mic_layouts.py。"
        )

    if not MIC_LAYOUTS_YAML.exists():
        raise FileNotFoundError(
            f"缺少 {MIC_LAYOUTS_YAML}，请先运行 scripts\\04_cage_mic_layouts.py。"
        )

    cage = load_yaml(CAGE_YAML)
    mic_layouts = load_yaml(MIC_LAYOUTS_YAML)

    check_baseline_cage(cage)

    mic_ids, mic_positions = get_mic_positions(mic_layouts, BASELINE_MIC_LAYOUT)
    source_position = build_source_position(cage)

    print(f"[INFO] 使用布局: {BASELINE_MIC_LAYOUT}")
    print(f"[INFO] 麦克风数量: {len(mic_ids)}")
    print(f"[INFO] 声源位置: {source_position.tolist()}")
    print("")

    source_signal = generate_probe_signal(fs=FS)

    multichannel, distances, delays_sec = build_anechoic_multichannel_scene(
        source_signal=source_signal,
        source_position=source_position,
        mic_positions=mic_positions,
        fs=FS,
        sound_speed=SPEED_OF_SOUND,
    )

    df, aux = build_pair_records(
        mic_ids=mic_ids,
        mic_positions=mic_positions,
        multichannel=multichannel,
        delays_sec=delays_sec,
        fs=FS,
    )

    summary = aux["summary"]
    gcc_cache = aux["gcc_cache"]

    df_out = df.copy()
    df_out = df_out.sort_values(
        by=["is_main_pair", "abs_error_samples"],
        ascending=[False, False],
    ).reset_index(drop=True)

    df_out.to_csv(TDOA_TABLE_CSV, index=False, encoding="utf-8-sig")

    plot_tdoa_error_curve(df_out, TDOA_CURVE_PNG)
    plot_main_pair_gcc_curve(df_out, gcc_cache, GCC_MAIN_PAIR_PNG)

    scene_config = {
        "task": "week2_day1_gcc_phat_tdoa_validation",
        "script": "scripts/day1.py",
        "scene_type": "anechoic_single_source_direct_path",
        "fs": FS,
        "speed_of_sound": SPEED_OF_SOUND,
        "cage_yaml": rel_to_root(CAGE_YAML),
        "mic_layouts_yaml": rel_to_root(MIC_LAYOUTS_YAML),
        "mic_layout": BASELINE_MIC_LAYOUT,
        "source_position": [round(float(v), 6) for v in source_position],
        "probe_signal": {
            "type": "deterministic_wideband_noise",
            "duration_sec": PROBE_SIGNAL_DURATION_SEC,
            "random_seed": PROBE_RANDOM_SEED,
        },
        "gcc_phat": {
            "interp": GCC_INTERP,
            "main_pair_rule": MAIN_PAIR_RULE,
            "main_pair_error_threshold_samples": MAIN_PAIR_ERROR_THRESHOLD_SAMPLES,
        },
        "outputs": {
            "tdoa_error_table": rel_to_root(TDOA_TABLE_CSV),
            "tdoa_error_curve": rel_to_root(TDOA_CURVE_PNG),
            "main_pair_gcc_curve": rel_to_root(GCC_MAIN_PAIR_PNG),
            "report": rel_to_root(REPORT_MD),
        },
        "note": "本脚本只完成 GCC-PHAT 与理论 TDOA 验证，不依赖 11_build_paper_sim_scenes.py。"
    }

    save_yaml(scene_config, SCENE_CONFIG_YAML)

    write_report(
        cage=cage,
        mic_ids=mic_ids,
        mic_positions=mic_positions,
        source_position=source_position,
        distances=distances,
        delays_sec=delays_sec,
        df=df_out,
        summary=summary,
    )

    print("========== 处理完成 ==========")
    print(f"主要麦对: {summary['main_pair']}")
    print(f"主要麦对绝对误差: {summary['main_pair_abs_error_samples']:.4f} samples")
    print(f"全部麦对最大绝对误差: {summary['all_pairs_max_abs_error_samples']:.4f} samples")
    print(f"TDOA误差表: {TDOA_TABLE_CSV}")
    print(f"TDOA误差曲线: {TDOA_CURVE_PNG}")
    print(f"主要麦对GCC曲线: {GCC_MAIN_PAIR_PNG}")
    print(f"场景配置: {SCENE_CONFIG_YAML}")
    print(f"报告: {REPORT_MD}")
    print("")

    if summary["main_pair_pass"]:
        print("[PASS] 今日验收通过：主要麦对 TDOA 误差 ≤ 1 个采样点。")
    else:
        print("[WARN] 今日验收未通过：主要麦对 TDOA 误差 > 1 个采样点，需要检查原因。")


if __name__ == "__main__":
    main()