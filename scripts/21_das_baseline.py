# -*- coding: utf-8 -*-
"""
第 3 周 Day1：DAS 基线（延迟求和波束形成）

目标：
1. 复用第 1 周的鸡笼坐标、麦克风布局和声学仿真思路；
2. 复用第 2 周 Day3 的 SRP-PHAT 单源定位函数，得到估计声源位置；
3. 分别使用真实声源位置和估计声源位置计算近场延迟；
4. 对多通道麦克风信号进行延迟对齐和叠加平均；
5. 保存单通道、真实位置 DAS、估计位置 DAS 的音频和频谱图；
6. 输出 das_results.csv 和 Markdown 报告。

运行：
    cd D:\\project\\chicken_acoustics
    python scripts\\21_das_baseline.py --n-scenes 3 --clean

正式小规模运行：
    python scripts\\21_das_baseline.py --n-scenes 20 --clean

说明：
- DAS = Delay-and-Sum，中文可称为“延迟求和波束形成”。
- 本脚本默认使用 mic_8、RT60=0.30 s、SNR=20 dB、采样率=48 kHz。
- 本脚本默认使用程序生成的宽带 probe 信号，便于稳定验证 DAS 流程。
- 如需使用真实鸡叫片段，可加参数：--source-mode real。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import librosa
except ImportError as exc:
    raise ImportError("当前脚本需要 librosa。请先安装 librosa。") from exc

try:
    import pyroomacoustics as pra
except ImportError as exc:
    raise ImportError("当前脚本需要 pyroomacoustics。请先安装 pyroomacoustics。") from exc

try:
    from scipy.signal import correlate, correlation_lags
except ImportError as exc:
    raise ImportError("当前脚本需要 scipy。请先安装 scipy。") from exc


# ============================================================
# 0. 路径与动态加载
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DAY3_SCRIPT = PROJECT_ROOT / "scripts" / "13_single_source_batch.py"

OUTPUT_DIR = PROJECT_ROOT / "results" / "week3" / "day1" / "das"
AUDIO_DIR = OUTPUT_DIR / "audio"
FIGURE_DIR = OUTPUT_DIR / "figures"
SCENE_DIR = OUTPUT_DIR / "scenes"

RESULT_CSV = OUTPUT_DIR / "das_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "das_summary.csv"
REPORT_MD = OUTPUT_DIR / "week3_day1_das_report.md"
CONFIG_YAML = OUTPUT_DIR / "das_config.yaml"

PROCESSED_SEGMENT_ROOT = PROJECT_ROOT / "data" / "processed_segments"


def load_day3_module():
    """动态加载第 2 周 Day3 单源定位脚本，复用已验证的 SRP-PHAT 函数。"""
    if not DAY3_SCRIPT.exists():
        raise FileNotFoundError(
            f"找不到第2周 Day3 脚本：{DAY3_SCRIPT}\n"
            "请确认文件存在：scripts\\13_single_source_batch.py"
        )

    spec = importlib.util.spec_from_file_location("day3_single_source_batch", DAY3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{DAY3_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["day3_single_source_batch"] = module
    spec.loader.exec_module(module)
    return module


day3 = load_day3_module()


# ============================================================
# 1. 默认参数
# ============================================================

FS = 48_000
SPEED_OF_SOUND = 343.0

DEFAULT_LAYOUT = "mic_8"
DEFAULT_RT60_SEC = 0.30
DEFAULT_SNR_DB = 20.0
DEFAULT_GRID_SPACING_M = 0.02

SOURCE_Z = 0.35
SEARCH_Z = 0.35

SOURCE_MARGIN_X = 0.08
SOURCE_MARGIN_Y = 0.08

PROBE_DURATION_SEC = 2.0
PROBE_RANDOM_SEED = 20260701

MAX_ORDER_CAP = 12
WAV_SUBTYPE = "FLOAT"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


# ============================================================
# 2. 通用工具函数
# ============================================================

def reset_output_dir(path: Path) -> None:
    """清空并重建输出目录。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs() -> None:
    """创建输出目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """保存 YAML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rel_to_root(path: Path) -> str:
    """返回相对于项目根目录的路径字符串。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def normalize_for_wav(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """仅用于保存 wav 的安全归一化，避免爆音。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs > 1e-12:
        x = x / max_abs * peak
    return x.astype(np.float32)


def save_mono_wav(path: Path, audio: np.ndarray, fs: int = FS) -> None:
    """保存单通道 wav。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, normalize_for_wav(audio), fs, subtype=WAV_SUBTYPE)


def save_multichannel_wav(path: Path, multichannel: np.ndarray, fs: int = FS) -> None:
    """保存多通道 wav。输入形状为 [M, N]，保存时转为 [N, M]。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(multichannel, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"多通道信号必须是二维数组，当前形状：{x.shape}")

    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs > 1e-12:
        x = x / max_abs * 0.95

    sf.write(path, x.T.astype(np.float32), fs, subtype=WAV_SUBTYPE)


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
# 3. 声源、场景与定位准备
# ============================================================

def sample_single_source_position(
    cage_dims: Dict[str, float],
    rng: np.random.Generator,
    source_z: float = SOURCE_Z,
) -> np.ndarray:
    """在鸡笼有效区域内随机采样一个单源位置。"""
    x = rng.uniform(SOURCE_MARGIN_X, cage_dims["length_x"] - SOURCE_MARGIN_X)
    y = rng.uniform(SOURCE_MARGIN_Y, cage_dims["width_y"] - SOURCE_MARGIN_Y)
    return np.array([x, y, source_z], dtype=np.float64)


def find_audio_segments(root: Path) -> List[Path]:
    """查找 processed_segments 下的真实音频片段。"""
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def load_real_source_segment(
    rng: np.random.Generator,
    fs: int,
    max_sec: float = 3.0,
) -> Tuple[np.ndarray, str]:
    """从 data/processed_segments 中随机读取一个真实鸡叫片段。"""
    files = find_audio_segments(PROCESSED_SEGMENT_ROOT)
    if not files:
        raise FileNotFoundError(
            f"没有找到真实片段：{PROCESSED_SEGMENT_ROOT}\n"
            "请先运行第1周 Day3 预处理脚本，或使用 --source-mode probe。"
        )

    path = files[int(rng.integers(0, len(files)))]
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y.astype(np.float64)

    if sr != fs:
        y = librosa.resample(y, orig_sr=sr, target_sr=fs).astype(np.float64)

    max_len = int(round(max_sec * fs))
    if len(y) > max_len:
        start = int(rng.integers(0, len(y) - max_len + 1))
        y = y[start:start + max_len]

    y = y - np.mean(y)
    y = y / (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float64), rel_to_root(path)


def make_source_signal(
    source_mode: str,
    fs: int,
    seed: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    """生成或读取声源信号。"""
    if source_mode == "real":
        return load_real_source_segment(rng=rng, fs=fs)

    if source_mode == "probe":
        signal = day3.make_probe_signal(
            fs=fs,
            duration_sec=PROBE_DURATION_SEC,
            seed=seed,
        )
        return signal.astype(np.float64), "generated_probe_signal"

    raise ValueError(f"未知 source_mode：{source_mode}，只能是 probe 或 real。")


def compute_room_material_and_order(room_dim: np.ndarray, rt60_sec: float) -> Tuple[float, int, int]:
    """根据目标 RT60 计算吸声系数和镜像源阶数。"""
    absorption, max_order_raw = pra.inverse_sabine(rt60_sec, room_dim)
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


def add_white_noise_at_snr(clean_signals: np.ndarray, snr_db: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """按指定 SNR 加白噪声，返回 noisy 和 noise。"""
    rng = np.random.default_rng(seed)
    clean_signals = np.asarray(clean_signals, dtype=np.float64)

    signal_power = float(np.mean(clean_signals ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    noise = rng.standard_normal(clean_signals.shape)
    noise = noise / (np.std(noise) + 1e-12)
    noise = noise * math.sqrt(noise_power)

    noisy = clean_signals + noise

    max_abs = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    if max_abs > 1.0:
        noisy = noisy / (max_abs + 1e-12)
        clean_signals = clean_signals / (max_abs + 1e-12)
        noise = noise / (max_abs + 1e-12)

    return noisy.astype(np.float64), noise.astype(np.float64)


def simulate_clean_noisy_scene(
    room_dim: np.ndarray,
    source_signal: np.ndarray,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    rt60_sec: float,
    snr_db: float,
    noise_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """
    使用 Pyroomacoustics 生成单源多通道场景。

    返回：
        clean: shape = [M, N]，无加性噪声的多通道接收信号
        noisy: shape = [M, N]，加噪后的多通道接收信号
        noise: shape = [M, N]，白噪声
        absorption, max_order_used, max_order_raw
    """
    absorption, max_order_used, max_order_raw = compute_room_material_and_order(room_dim, rt60_sec)

    try:
        room = pra.ShoeBox(
            p=room_dim,
            fs=fs,
            materials=make_pra_material(absorption),
            max_order=max_order_used,
            air_absorption=False,
        )
    except TypeError:
        room = pra.ShoeBox(
            p=room_dim,
            fs=fs,
            materials=make_pra_material(absorption),
            max_order=max_order_used,
        )

    room.add_source(source_position, signal=source_signal)

    mic_array = pra.MicrophoneArray(mic_positions.T, fs=fs)
    room.add_microphone_array(mic_array)

    room.simulate()

    clean = room.mic_array.signals.astype(np.float64)
    clean = clean - np.mean(clean, axis=1, keepdims=True)
    clean = clean / (np.max(np.abs(clean)) + 1e-12)

    noisy, noise = add_white_noise_at_snr(
        clean_signals=clean,
        snr_db=snr_db,
        seed=noise_seed,
    )

    return clean, noisy, noise, absorption, max_order_used, max_order_raw


def prepare_localizer(
    cage_dims: Dict[str, float],
    mic_positions: np.ndarray,
    grid_spacing_m: float,
) -> Dict[str, Any]:
    """准备 SRP-PHAT 定位所需的网格、麦克风对和 TDOA 缓存。"""
    x_values, y_values, grid_points = day3.make_grid(
        length_x=float(cage_dims["length_x"]),
        width_y=float(cage_dims["width_y"]),
        spacing=grid_spacing_m,
        margin_x=SOURCE_MARGIN_X,
        margin_y=SOURCE_MARGIN_Y,
        z=SEARCH_Z,
    )

    grid_shape = (len(y_values), len(x_values))
    pairs = day3.build_mic_pairs(len(mic_positions))

    max_tau_samples = day3.compute_max_tau_samples(
        mic_positions=mic_positions,
        pairs=pairs,
        fs=FS,
        sound_speed=SPEED_OF_SOUND,
    )

    baseline_weights = day3.compute_baseline_weights(
        mic_positions=mic_positions,
        pairs=pairs,
    )

    grid_tdoa_cache = day3.precompute_grid_tdoa_samples(
        grid_points=grid_points,
        mic_positions=mic_positions,
        pairs=pairs,
        fs=FS,
        sound_speed=SPEED_OF_SOUND,
    )

    return {
        "x_values": x_values,
        "y_values": y_values,
        "grid_points": grid_points,
        "grid_shape": grid_shape,
        "pairs": pairs,
        "max_tau_samples": max_tau_samples,
        "baseline_weights": baseline_weights,
        "grid_tdoa_cache": grid_tdoa_cache,
    }


def estimate_position_srp(multichannel: np.ndarray, localizer: Dict[str, Any]) -> Tuple[np.ndarray, float, np.ndarray]:
    """调用第2周 Day3 的 SRP-PHAT 单源定位函数，得到估计位置。"""
    pred_position, peak_score, score_values = day3.srp_phat_localize_one_scene(
        multichannel=multichannel,
        grid_points=localizer["grid_points"],
        grid_shape=localizer["grid_shape"],
        pairs=localizer["pairs"],
        grid_tdoa_cache=localizer["grid_tdoa_cache"],
        baseline_weights=localizer["baseline_weights"],
        interp=day3.INTERP,
        max_tau_samples=localizer["max_tau_samples"],
    )
    return pred_position.astype(np.float64), float(peak_score), score_values


# ============================================================
# 4. DAS 延迟求和波束形成
# ============================================================

def compute_nearfield_delays(
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    sound_speed: float = SPEED_OF_SOUND,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据声源位置和麦克风位置计算近场传播时间。

    返回：
        distances: 每个麦克风到声源的距离，单位 m
        delays_sec: 每个麦克风对应的传播时间，单位 s
    """
    source_position = np.asarray(source_position, dtype=np.float64)
    mic_positions = np.asarray(mic_positions, dtype=np.float64)

    distances = np.linalg.norm(mic_positions - source_position[None, :], axis=1)
    delays_sec = distances / sound_speed

    return distances.astype(np.float64), delays_sec.astype(np.float64)


def fractional_advance(signal: np.ndarray, advance_samples: float) -> np.ndarray:
    """
    将信号向前移动 advance_samples 个采样点。

    说明：
    - 某个麦克风收到目标声源越晚，advance_samples 越大；
    - 用线性插值实现小数采样点级对齐；
    - 输出长度与输入相同，超出范围补 0。
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return signal

    sample_index = np.arange(n, dtype=np.float64)
    query_index = sample_index + float(advance_samples)

    aligned = np.interp(
        query_index,
        sample_index,
        signal,
        left=0.0,
        right=0.0,
    )
    return aligned.astype(np.float64)


def das_beamform(
    multichannel: np.ndarray,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    对指定声源位置执行 DAS。

    输入：
        multichannel: shape = [M, N]
        source_position: shape = [3]
        mic_positions: shape = [M, 3]

    返回：
        enhanced: DAS 输出，shape = [N]
        relative_delays_samples: 相对最早到达麦克风的延迟，单位 samples
        distances: 声源到每个麦克风的距离，单位 m
        ref_mic_index: 最早到达的参考麦克风编号
    """
    x = np.asarray(multichannel, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"multichannel 必须是 [M, N]，当前形状：{x.shape}")

    distances, delays_sec = compute_nearfield_delays(
        source_position=source_position,
        mic_positions=mic_positions,
    )

    min_delay = float(np.min(delays_sec))
    ref_mic_index = int(np.argmin(delays_sec))

    relative_delays_samples = (delays_sec - min_delay) * fs

    aligned_channels = []
    for mic_idx in range(x.shape[0]):
        aligned = fractional_advance(
            signal=x[mic_idx],
            advance_samples=float(relative_delays_samples[mic_idx]),
        )
        aligned_channels.append(aligned)

    aligned_stack = np.stack(aligned_channels, axis=0)
    enhanced = np.mean(aligned_stack, axis=0)
    enhanced = enhanced - np.mean(enhanced)

    return enhanced.astype(np.float64), relative_delays_samples.astype(np.float64), distances, ref_mic_index


# ============================================================
# 5. 指标计算与可视化
# ============================================================

def align_to_reference_by_corr(
    estimated: np.ndarray,
    reference: np.ndarray,
    fs: int,
    max_lag_sec: float = 0.10,
) -> Tuple[np.ndarray, int]:
    """
    用互相关在有限范围内对齐 estimated 和 reference。

    返回：
        aligned_estimated: 对齐后的 estimated
        best_lag: 最佳整数延迟，单位 samples
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    n = min(len(estimated), len(reference))
    if n <= 1:
        return estimated[:n], 0

    est = estimated[:n] - np.mean(estimated[:n])
    ref = reference[:n] - np.mean(reference[:n])

    corr = correlate(est, ref, mode="full", method="fft")
    lags = correlation_lags(len(est), len(ref), mode="full")

    max_lag = int(round(max_lag_sec * fs))
    mask = np.abs(lags) <= max_lag

    if not np.any(mask):
        return est, 0

    best_lag = int(lags[mask][np.argmax(np.abs(corr[mask]))])

    aligned = np.zeros_like(est)
    if best_lag > 0:
        # estimated 相对 reference 偏晚，向前移
        aligned[:n - best_lag] = est[best_lag:n]
    elif best_lag < 0:
        # estimated 相对 reference 偏早，向后移
        shift = -best_lag
        aligned[shift:n] = est[:n - shift]
    else:
        aligned = est.copy()

    return aligned.astype(np.float64), best_lag


def si_sdr_db(estimated: np.ndarray, reference: np.ndarray, fs: int) -> Tuple[float, int]:
    """计算自动对齐后的 SI-SDR，返回 SI-SDR 和对齐 lag。"""
    aligned, lag = align_to_reference_by_corr(estimated, reference, fs)
    ref = np.asarray(reference[:len(aligned)], dtype=np.float64)
    ref = ref - np.mean(ref)

    if len(aligned) == 0 or np.sum(ref ** 2) < 1e-12:
        return float("nan"), lag

    scale = float(np.dot(aligned, ref) / (np.dot(ref, ref) + 1e-12))
    target = scale * ref
    error = aligned - target

    value = 10.0 * math.log10((np.sum(target ** 2) + 1e-12) / (np.sum(error ** 2) + 1e-12))
    return float(value), lag


def snr_against_reference_db(estimated: np.ndarray, reference: np.ndarray, fs: int) -> Tuple[float, int]:
    """计算自动对齐后的参考 SNR。"""
    aligned, lag = align_to_reference_by_corr(estimated, reference, fs)
    ref = np.asarray(reference[:len(aligned)], dtype=np.float64)
    ref = ref - np.mean(ref)

    if len(aligned) == 0 or np.mean(ref ** 2) < 1e-12:
        return float("nan"), lag

    err = aligned - ref
    value = 10.0 * math.log10((np.mean(ref ** 2) + 1e-12) / (np.mean(err ** 2) + 1e-12))
    return float(value), lag


def plot_spectrogram(audio: np.ndarray, fs: int, out_path: Path, title: str) -> None:
    """保存单个频谱图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float64)
    audio = audio - np.mean(audio)

    plt.figure(figsize=(9, 4))
    plt.specgram(audio, NFFT=1024, Fs=fs, noverlap=512)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Power")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_spectrograms(
    mic0: np.ndarray,
    das_true: np.ndarray,
    das_est: np.ndarray,
    fs: int,
    out_path: Path,
    title: str,
) -> None:
    """保存三种方法的频谱对比图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    signals = [mic0, das_true, das_est]
    titles = ["Single microphone", "DAS with true position", "DAS with estimated position"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, sig, sub_title in zip(axes, signals, titles):
        sig = np.asarray(sig, dtype=np.float64)
        sig = sig - np.mean(sig)
        ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=512)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(sub_title)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# 6. 单场景实验
# ============================================================

def run_one_scene(
    scene_index: int,
    cage_dims: Dict[str, float],
    room_dim: np.ndarray,
    mic_positions: np.ndarray,
    localizer: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """生成一个单源场景，执行真实位置 DAS 和估计位置 DAS。"""
    rng = np.random.default_rng(args.seed + scene_index)

    scene_name = f"scene_{scene_index:03d}"
    scene_dir = SCENE_DIR / scene_name
    scene_audio_dir = AUDIO_DIR / scene_name
    scene_figure_dir = FIGURE_DIR / scene_name

    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_audio_dir.mkdir(parents=True, exist_ok=True)
    scene_figure_dir.mkdir(parents=True, exist_ok=True)

    true_position = sample_single_source_position(
        cage_dims=cage_dims,
        rng=rng,
        source_z=SOURCE_Z,
    )

    source_signal, source_ref = make_source_signal(
        source_mode=args.source_mode,
        fs=FS,
        seed=args.seed + 10000 + scene_index,
        rng=rng,
    )

    clean, noisy, noise, absorption, max_order_used, max_order_raw = simulate_clean_noisy_scene(
        room_dim=room_dim,
        source_signal=source_signal,
        source_position=true_position,
        mic_positions=mic_positions,
        fs=FS,
        rt60_sec=args.rt60,
        snr_db=args.snr,
        noise_seed=args.seed + 20000 + scene_index,
    )

    est_position, peak_score, _ = estimate_position_srp(
        multichannel=noisy,
        localizer=localizer,
    )

    localization_error_m = float(np.linalg.norm(est_position[:2] - true_position[:2]))

    das_true, true_rel_delays, true_distances, true_ref_mic = das_beamform(
        multichannel=noisy,
        source_position=true_position,
        mic_positions=mic_positions,
        fs=FS,
    )

    das_est, est_rel_delays, est_distances, est_ref_mic = das_beamform(
        multichannel=noisy,
        source_position=est_position,
        mic_positions=mic_positions,
        fs=FS,
    )

    # ------------------------------------------------------------
    # 关键修正：DAS 输出不能直接和第 0 个麦克风的 clean 信号比较。
    # DAS(noisy) 的参考应该是 DAS(clean)，因为 DAS 输出是多通道对齐叠加后的信号，
    # 它已经不是某一个单独麦克风通道的波形。
    # ------------------------------------------------------------
    das_true_clean, _, _, _ = das_beamform(
        multichannel=clean,
        source_position=true_position,
        mic_positions=mic_positions,
        fs=FS,
    )

    das_est_clean, _, _, _ = das_beamform(
        multichannel=clean,
        source_position=est_position,
        mic_positions=mic_positions,
        fs=FS,
    )

    mic0 = noisy[0]
    clean_mic0_ref = clean[0]

    # 单通道：noisy mic0 与 clean mic0 比较。
    mic0_sisdr, mic0_lag = si_sdr_db(mic0, clean_mic0_ref, FS)
    mic0_snr, _ = snr_against_reference_db(mic0, clean_mic0_ref, FS)

    # DAS：DAS(noisy) 与 DAS(clean) 比较。
    true_sisdr, true_lag = si_sdr_db(das_true, das_true_clean, FS)
    est_sisdr, est_lag = si_sdr_db(das_est, das_est_clean, FS)

    true_snr, _ = snr_against_reference_db(das_true, das_true_clean, FS)
    est_snr, _ = snr_against_reference_db(das_est, das_est_clean, FS)

    # 保存音频
    mixture_path = scene_audio_dir / "mixture_multichannel.wav"
    clean_path = scene_audio_dir / "clean_multichannel.wav"
    noise_path = scene_audio_dir / "noise_multichannel.wav"
    source_path = scene_audio_dir / "source_signal.wav"
    clean_ref_path = scene_audio_dir / "target_clean_mic0.wav"
    mic0_path = scene_audio_dir / "mic0_single.wav"
    das_true_path = scene_audio_dir / "das_true_position.wav"
    das_est_path = scene_audio_dir / "das_est_position.wav"
    das_true_clean_path = scene_audio_dir / "das_true_position_clean_reference.wav"
    das_est_clean_path = scene_audio_dir / "das_est_position_clean_reference.wav"

    save_multichannel_wav(mixture_path, noisy, FS)
    save_multichannel_wav(clean_path, clean, FS)
    save_multichannel_wav(noise_path, noise, FS)
    save_mono_wav(source_path, source_signal, FS)
    save_mono_wav(clean_ref_path, clean_mic0_ref, FS)
    save_mono_wav(mic0_path, mic0, FS)
    save_mono_wav(das_true_path, das_true, FS)
    save_mono_wav(das_est_path, das_est, FS)
    save_mono_wav(das_true_clean_path, das_true_clean, FS)
    save_mono_wav(das_est_clean_path, das_est_clean, FS)

    # 保存频谱图
    plot_spectrogram(
        audio=mic0,
        fs=FS,
        out_path=scene_figure_dir / "spectrogram_mic0_single.png",
        title=f"{scene_name} - single microphone",
    )
    plot_spectrogram(
        audio=das_true,
        fs=FS,
        out_path=scene_figure_dir / "spectrogram_das_true_position.png",
        title=f"{scene_name} - DAS with true position",
    )
    plot_spectrogram(
        audio=das_est,
        fs=FS,
        out_path=scene_figure_dir / "spectrogram_das_est_position.png",
        title=f"{scene_name} - DAS with estimated position",
    )
    plot_compare_spectrograms(
        mic0=mic0,
        das_true=das_true,
        das_est=das_est,
        fs=FS,
        out_path=scene_figure_dir / "spectrogram_compare.png",
        title=f"{scene_name} - DAS baseline comparison",
    )

    metadata = {
        "scene_index": int(scene_index),
        "scene_name": scene_name,
        "fs": int(FS),
        "layout": args.layout,
        "num_mics": int(len(mic_positions)),
        "rt60_sec": float(args.rt60),
        "snr_db": float(args.snr),
        "source_mode": args.source_mode,
        "source_ref": source_ref,
        "true_source_position": [float(v) for v in true_position],
        "estimated_source_position": [float(v) for v in est_position],
        "localization_error_m": localization_error_m,
        "peak_score": float(peak_score),
        "absorption": float(absorption),
        "max_order_used": int(max_order_used),
        "max_order_raw": int(max_order_raw),
        "true_ref_mic_index": int(true_ref_mic),
        "est_ref_mic_index": int(est_ref_mic),
        "true_relative_delays_samples": [float(v) for v in true_rel_delays],
        "est_relative_delays_samples": [float(v) for v in est_rel_delays],
        "true_distances_m": [float(v) for v in true_distances],
        "est_distances_m": [float(v) for v in est_distances],
    }
    save_json(metadata, scene_dir / "metadata.json")

    return {
        "scene_index": int(scene_index),
        "layout": args.layout,
        "num_mics": int(len(mic_positions)),
        "rt60_sec": float(args.rt60),
        "snr_db": float(args.snr),
        "source_mode": args.source_mode,
        "true_x": float(true_position[0]),
        "true_y": float(true_position[1]),
        "true_z": float(true_position[2]),
        "est_x": float(est_position[0]),
        "est_y": float(est_position[1]),
        "est_z": float(est_position[2]),
        "localization_error_m": localization_error_m,
        "localization_error_cm": localization_error_m * 100.0,
        "srp_peak_score": float(peak_score),
        "mic0_si_sdr_db": float(mic0_sisdr),
        "das_true_si_sdr_db": float(true_sisdr),
        "das_est_si_sdr_db": float(est_sisdr),
        "das_true_si_sdri_db": float(true_sisdr - mic0_sisdr),
        "das_est_si_sdri_db": float(est_sisdr - mic0_sisdr),
        "mic0_snr_ref_db": float(mic0_snr),
        "das_true_snr_ref_db": float(true_snr),
        "das_est_snr_ref_db": float(est_snr),
        "das_true_snr_improvement_db": float(true_snr - mic0_snr),
        "das_est_snr_improvement_db": float(est_snr - mic0_snr),
        "mic0_align_lag_samples": int(mic0_lag),
        "das_true_align_lag_samples": int(true_lag),
        "das_est_align_lag_samples": int(est_lag),
        "mic0_audio": rel_to_root(mic0_path),
        "das_true_audio": rel_to_root(das_true_path),
        "das_est_audio": rel_to_root(das_est_path),
        "das_true_clean_reference_audio": rel_to_root(das_true_clean_path),
        "das_est_clean_reference_audio": rel_to_root(das_est_clean_path),
        "compare_figure": rel_to_root(scene_figure_dir / "spectrogram_compare.png"),
        "metadata": rel_to_root(scene_dir / "metadata.json"),
    }


# ============================================================
# 7. 汇总、报告与主函数
# ============================================================

def build_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """生成 DAS 汇总表。"""
    if result_df.empty:
        return pd.DataFrame()

    rows = []
    metrics = [
        "localization_error_cm",
        "mic0_si_sdr_db",
        "das_true_si_sdr_db",
        "das_est_si_sdr_db",
        "das_true_si_sdri_db",
        "das_est_si_sdri_db",
        "mic0_snr_ref_db",
        "das_true_snr_ref_db",
        "das_est_snr_ref_db",
        "das_true_snr_improvement_db",
        "das_est_snr_improvement_db",
    ]

    for metric in metrics:
        values = pd.to_numeric(result_df[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            rows.append({
                "metric": metric,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
        else:
            rows.append({
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })

    return pd.DataFrame(rows)


def write_report(result_df: pd.DataFrame, summary_df: pd.DataFrame, args: argparse.Namespace) -> None:
    """生成 Markdown 报告。"""
    pass_true = False
    pass_est = False

    if not result_df.empty:
        true_mean = float(result_df["das_true_si_sdri_db"].mean())
        est_mean = float(result_df["das_est_si_sdri_db"].mean())
        pass_true = true_mean > 0.0
        pass_est = est_mean > 0.0
    else:
        true_mean = float("nan")
        est_mean = float("nan")

    lines = []
    lines.append("# 第3周 Day1：DAS 基线实验报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("本实验实现 DAS（延迟求和波束形成）基线方法。")
    lines.append("给定声源位置和麦克风坐标，先计算声源到各麦克风的近场传播延迟，")
    lines.append("再将多通道信号按延迟对齐并叠加平均，得到增强后的目标声音。")
    lines.append("")
    lines.append("本实验分别使用两类位置：")
    lines.append("")
    lines.append("- 真实声源位置：用于验证 DAS 代码本身是否正确；")
    lines.append("- 第2周 SRP-PHAT 估计位置：用于验证定位结果是否能用于后续声音增强。")
    lines.append("")
    lines.append("评价时，单通道输出与 clean mic0 比较；DAS 输出与同一位置下的 clean DAS 参考比较，")
    lines.append("避免把多通道对齐叠加后的 DAS 输出错误地拿去和单个麦克风 clean 信号比较。")
    lines.append("")
    lines.append("## 2. 实验配置")
    lines.append("")
    lines.append(f"- 场景数：{args.n_scenes}")
    lines.append(f"- 麦克风布局：`{args.layout}`")
    lines.append(f"- 采样率：{FS} Hz")
    lines.append(f"- RT60：{args.rt60:.2f} s")
    lines.append(f"- SNR：{args.snr:.1f} dB")
    lines.append(f"- 声源模式：`{args.source_mode}`")
    lines.append(f"- 搜索网格间距：{args.grid_spacing:.2f} m")
    lines.append("")
    lines.append("## 3. 汇总结果")
    lines.append("")
    lines.append(dataframe_to_markdown(summary_df))
    lines.append("")
    lines.append("## 4. 验收判断")
    lines.append("")

    if pass_true:
        lines.append(f"- 真实位置 DAS 平均 SI-SDRi = {true_mean:.2f} dB，大于 0 dB，说明真实位置 DAS 优于单通道。")
    else:
        lines.append(f"- 真实位置 DAS 平均 SI-SDRi = {true_mean:.2f} dB，未明显优于单通道，需要检查延迟方向、坐标顺序或评价对齐。")

    if pass_est:
        lines.append(f"- 估计位置 DAS 平均 SI-SDRi = {est_mean:.2f} dB，大于 0 dB，说明第2周定位结果可以用于 DAS 增强。")
    else:
        lines.append(f"- 估计位置 DAS 平均 SI-SDRi = {est_mean:.2f} dB，未明显优于单通道，可能受定位误差影响。")

    lines.append("")
    lines.append("## 5. 输出文件")
    lines.append("")
    lines.append(f"- 逐场景结果：`{rel_to_root(RESULT_CSV)}`")
    lines.append(f"- 汇总结果：`{rel_to_root(SUMMARY_CSV)}`")
    lines.append(f"- 音频目录：`{rel_to_root(AUDIO_DIR)}`")
    lines.append(f"- 频谱图目录：`{rel_to_root(FIGURE_DIR)}`")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第3周 Day1：DAS 基线实验")
    parser.add_argument("--n-scenes", type=int, default=3, help="生成并测试的单源场景数量")
    parser.add_argument("--layout", type=str, default=DEFAULT_LAYOUT, help="麦克风布局名称，例如 mic_8")
    parser.add_argument("--rt60", type=float, default=DEFAULT_RT60_SEC, help="RT60，单位秒")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR_DB, help="加性白噪声 SNR，单位 dB")
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING_M, help="SRP-PHAT 搜索网格间距，单位 m")
    parser.add_argument("--source-mode", type=str, default="probe", choices=["probe", "real"], help="声源类型：probe=宽带探针信号，real=真实鸡叫片段")
    parser.add_argument("--seed", type=int, default=20260701, help="随机种子")
    parser.add_argument("--clean", action="store_true", help="运行前清空旧输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean:
        reset_output_dir(OUTPUT_DIR)
    else:
        ensure_output_dirs()

    config = {
        "task": "week3_day1_das_baseline",
        "n_scenes": int(args.n_scenes),
        "layout": args.layout,
        "fs": int(FS),
        "rt60_sec": float(args.rt60),
        "snr_db": float(args.snr),
        "grid_spacing_m": float(args.grid_spacing),
        "source_mode": args.source_mode,
        "seed": int(args.seed),
        "source_z": float(SOURCE_Z),
        "search_z": float(SEARCH_Z),
    }
    save_yaml(config, CONFIG_YAML)

    cage_cfg = day3.load_yaml(day3.CAGE_YAML)
    mic_cfg = day3.load_yaml(day3.MIC_LAYOUTS_YAML)

    cage_dims = day3.get_cage_dimensions(cage_cfg)
    room_dim = np.array(
        [
            float(cage_dims["length_x"]),
            float(cage_dims["width_y"]),
            float(cage_dims["height_z"]),
        ],
        dtype=np.float64,
    )

    mic_df = day3.extract_mic_layout(mic_cfg, args.layout)
    mic_positions = mic_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    localizer = prepare_localizer(
        cage_dims=cage_dims,
        mic_positions=mic_positions,
        grid_spacing_m=args.grid_spacing,
    )

    print("========== 第3周 Day1：DAS 基线 ==========")
    print(f"[INFO] 项目根目录: {PROJECT_ROOT}")
    print(f"[INFO] 输出目录: {OUTPUT_DIR}")
    print(f"[INFO] 场景数: {args.n_scenes}")
    print(f"[INFO] 麦克风布局: {args.layout}, 麦克风数: {len(mic_positions)}")
    print(f"[INFO] RT60: {args.rt60:.2f} s, SNR: {args.snr:.1f} dB")
    print(f"[INFO] 声源模式: {args.source_mode}")
    print("")

    records: List[Dict[str, Any]] = []

    for scene_index in range(args.n_scenes):
        print(f"[INFO] 正在处理 scene_{scene_index:03d} ...")
        record = run_one_scene(
            scene_index=scene_index,
            cage_dims=cage_dims,
            room_dim=room_dim,
            mic_positions=mic_positions,
            localizer=localizer,
            args=args,
        )
        records.append(record)

        print(
            "       "
            f"定位误差={record['localization_error_cm']:.2f} cm, "
            f"真实DAS SI-SDRi={record['das_true_si_sdri_db']:.2f} dB, "
            f"估计DAS SI-SDRi={record['das_est_si_sdri_db']:.2f} dB"
        )

    result_df = pd.DataFrame(records)
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    summary_df = build_summary(result_df)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    write_report(result_df, summary_df, args)

    true_mean = float(result_df["das_true_si_sdri_db"].mean()) if not result_df.empty else float("nan")
    est_mean = float(result_df["das_est_si_sdri_db"].mean()) if not result_df.empty else float("nan")

    print("")
    print("========== 处理完成 ==========")
    print(f"逐场景结果: {RESULT_CSV}")
    print(f"汇总结果: {SUMMARY_CSV}")
    print(f"报告: {REPORT_MD}")
    print(f"音频目录: {AUDIO_DIR}")
    print(f"频谱图目录: {FIGURE_DIR}")
    print(f"真实位置 DAS 平均 SI-SDRi: {true_mean:.2f} dB")
    print(f"估计位置 DAS 平均 SI-SDRi: {est_mean:.2f} dB")

    if true_mean > 0.0:
        print("[PASS] 真实位置 DAS 优于单通道，达到 Day1 基线验收目标。")
    else:
        print("[WARN] 真实位置 DAS 未明显优于单通道，请检查延迟方向、坐标顺序或评价指标。")


if __name__ == "__main__":
    main()
