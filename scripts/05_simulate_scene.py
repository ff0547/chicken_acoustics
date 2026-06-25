# -*- coding: utf-8 -*-
"""
第 1 周第 5 天：单源和多源仿真器

当前阶段定位：
- 这是第 1 周的仿真器功能验证脚本；
- 默认生成 20 个测试场景；
- 目标是证明 RIR、source image、mixed audio、随机种子复现和求和验证都正常；
- 同时为后续论文级实验矩阵预留字段。

后续论文级实验：
- 不在本脚本中直接完成 E1～E10；
- 后续由 06_build_paper_sim_scenes.py 扩展每组 50 / 100 场景；
- 后续定位评价脚本可直接读取本脚本输出的 ground truth 坐标、SNR、RT60、麦克风布局等信息。

功能：
1. 读取第 4 天生成的 cage.yaml 和 mic_layouts.yaml；
2. 从 data/processed_segments 中随机选择真实音频片段作为声源；
3. 使用 Pyroomacoustics 生成 RIR 和 image-source 模型信息；
4. 支持单源 / 多源场景；
5. 支持声源起始时间、增益、RT60、SNR 和随机种子；
6. 生成 20 个可复现场景；
7. 保存每个声源到每个麦克风的 source image 音频；
8. 保存每个麦克风接收到的 mixed clean / mixed noisy / noise；
9. 检查 mixed clean == sum(source_images)；
10. 检查 mixed noisy == sum(source_images) + noise；
11. 输出带论文级元数据的 scene_manifest.csv 和 source_manifest.csv。
"""

from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import warnings

import numpy as np
import pandas as pd
import yaml
import soundfile as sf
import librosa

from scipy.signal import fftconvolve

import pyroomacoustics as pra


# =========================
# 路径配置
# =========================

ROOT = Path(__file__).resolve().parents[1]

CAGE_YAML = ROOT / "configs" / "week1" / "day4" / "cage.yaml"
MIC_LAYOUTS_YAML = ROOT / "configs" / "week1" / "day4" / "mic_layouts.yaml"

INPUT_SEGMENT_ROOT = ROOT / "data" / "processed_segments"

CONFIG_DIR = ROOT / "configs" / "week1" / "day5"
RESULT_DIR = ROOT / "results" / "week1" / "day5" / "simulate_scene"
SIM_DATA_ROOT = ROOT / "data" / "simulated_scenes" / "week1" / "day5"

SCENE_CONFIG_YAML = CONFIG_DIR / "sim_scenes.yaml"
SCENE_MANIFEST_CSV = RESULT_DIR / "scene_manifest.csv"
SOURCE_MANIFEST_CSV = RESULT_DIR / "source_manifest.csv"
VALIDATION_CSV = RESULT_DIR / "simulation_validation.csv"
QUALITY_REPORT_MD = RESULT_DIR / "simulation_quality_report.md"


# =========================
# 全局配置
# =========================

TARGET_SR = 48000

# 第 1 周第 5 天只做 20 个功能验证场景，不做每组100的正式实验矩阵
NUM_SCENES = 20

GLOBAL_SEED = 20260625

# 是否清空旧输出，避免重复运行时残留旧场景
CLEAR_OLD_OUTPUTS = True

# 场景最长音频时长，单位秒
SCENE_DURATION_SEC = 7.0

# 声源音频最多使用 5 秒
MAX_SOURCE_AUDIO_SEC = 5.0

# 声源起始时间范围
SOURCE_DELAY_RANGE_SEC = [0.0, 1.5]

# 声源增益范围
SOURCE_GAIN_RANGE = [0.60, 1.20]

# =========================
# 论文级基准场景参数
# =========================

# 基准麦克风布局：8 麦，鸡笼四周分布式
BASELINE_MIC_LAYOUT = "mic_8"

# 第1周 Day5 功能验证默认使用基准布局。
# 后续 E1 才正式展开 4/6/8/12。
LAYOUT_CHOICES_FOR_DAY5 = ["mic_8"]

# 如果你想在 Day5 顺手测试 4/6/8/12 是否都能仿真，可以改成：
# LAYOUT_CHOICES_FOR_DAY5 = ["mic_4", "mic_6", "mic_8", "mic_12"]

# 发声平面：基准 z=0.35 m
SOURCE_Z_BASELINE = 0.35

# 后续高度误差实验值：E7 或高度敏感性实验用
SOURCE_Z_VALUES = [0.25, 0.35, 0.45]

# Day5 功能验证固定在基准发声平面
SOURCE_Z_RANGE = [0.35, 0.35]

# 基准 RT60：0.30 s
BASELINE_RT60_SEC = 0.30
RT60_VALUES_SEC = [0.10, 0.30, 0.50, 0.70]

# Day5 功能验证固定基准 RT60
RT60_RANGE_SEC = [0.30, 0.30]

# 基准 SNR：20 dB
BASELINE_SNR_DB = 20.0
SNR_VALUES_DB = [0.0, 5.0, 10.0, 20.0, 30.0]

# Day5 功能验证固定基准 SNR
SNR_RANGE_DB = [20.0, 20.0]

# 活动源数：基准为 1 或 2；后续 E8 再正式展开 1/2/3
NUM_SOURCES_CHOICES = [1, 2]

# 新鸡笼尺寸为 1.20 × 0.75 × 0.60 m，边界留距不能太大
SOURCE_MARGIN_X = 0.08
SOURCE_MARGIN_Y = 0.08

# 输出 wav 使用 float，保证 source image 求和验证更精确
WAV_SUBTYPE = "FLOAT"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}

# 为后续论文级实验预留的元信息
RUN_MODE = "week1_day5_function_check"
PAPER_READY_METADATA = True

PLANNED_NEXT_EXPERIMENTS = {
    "E1": {
        "task": "single_source_localization",
        "variable": "num_mics",
        "values": ["mic_4", "mic_6", "mic_8", "mic_12"],
        "paper_sample_size_per_group": 100,
        "metrics": ["MAE_cm", "P90_cm", "Hit@10cm"],
    },
    "E2": {
        "task": "single_source_localization",
        "variable": "layout",
        "values": [
            "layout_corners_8",
            "layout_sides_8",
            "layout_single_side_8",
            "layout_compact_8",
        ],
        "paper_sample_size_per_group": 100,
        "metrics": ["error_cm", "runtime_sec"],
    },
    "E3": {
        "task": "single_source_localization",
        "variable": "snr_db",
        "values": [0, 5, 10, 20, 30],
        "paper_sample_size_per_group": 100,
        "metrics": ["error_cm", "miss_rate"],
    },
    "E4": {
        "task": "dual_source_localization",
        "variable": "source_distance_cm",
        "values": [10, 20, 30, 50, 80],
        "paper_sample_size_per_group": 100,
        "metrics": ["dual_source_hit_rate", "false_alarm_rate"],
    },
    "E5": {
        "task": "localization_rt60",
        "variable": "rt60_sec",
        "values": [0.10, 0.30, 0.50, 0.70],
        "paper_sample_size_per_group": 100,
        "metrics": ["error_cm", "peak_ratio"],
    },
    "E6": {
        "task": "reconstruction",
        "variable": "method",
        "values": ["single_channel", "DAS", "MVDR", "LCMV"],
        "paper_sample_size_per_group": 100,
        "metrics": ["SI_SDRi", "LogMel_distance"],
    },
    "E7": {
        "task": "reconstruction_position_error",
        "variable": "position_error_cm",
        "values": [0, 5, 10, 15],
        "paper_sample_size_per_group": 100,
        "metrics": ["SI_SDRi_drop"],
    },
    "E8": {
        "task": "joint_localization_and_separation",
        "variable": "num_active_sources",
        "values": [1, 2, 3],
        "paper_sample_size_per_group": 100,
        "metrics": ["joint_success_rate"],
    },
    "E9": {
        "task": "deep_method",
        "variable": "position_encoding",
        "values": ["none", "with_position", "position_perturbation"],
        "paper_sample_size_per_group": "unified_test_set",
        "metrics": ["SI_SDRi", "spectral_error"],
    },
    "E10": {
        "task": "complexity",
        "variable": "grid_and_num_mics",
        "values": "multiple_combinations",
        "paper_sample_size_per_group": "unified_hardware",
        "metrics": ["real_time_factor", "memory_mb"],
    },
}


# =========================
# 通用工具函数
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


def reset_output_dirs():
    """清空旧输出目录。"""
    for p in [CONFIG_DIR, RESULT_DIR, SIM_DATA_ROOT]:
        if p.exists():
            shutil.rmtree(p)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs():
    """创建输出目录。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def find_audio_segments() -> List[Path]:
    """查找可用于仿真的 1～5 秒音频片段。"""
    if not INPUT_SEGMENT_ROOT.exists():
        return []

    files = []
    for p in INPUT_SEGMENT_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)

    return sorted(files)


def load_audio_mono(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    读取单声道音频，并重采样到目标采样率。

    processed_segments 理论上已经是 48 kHz 单声道，
    这里仍保留检查，增强鲁棒性。
    """
    y, sr = librosa.load(path, sr=None, mono=True)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    y = y.astype(np.float32)

    max_samples = int(MAX_SOURCE_AUDIO_SEC * target_sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    # 去除极小直流偏置
    if len(y) > 0:
        y = y - np.mean(y)

    return y.astype(np.float32)


def normalize_peak(y: np.ndarray, target_peak: float = 0.80) -> np.ndarray:
    """将音频峰值归一化到 target_peak。"""
    if len(y) == 0:
        return y.astype(np.float32)

    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y.astype(np.float32)

    return (y / peak * target_peak).astype(np.float32)


def pad_or_crop(y: np.ndarray, length: int) -> np.ndarray:
    """将音频补零或截断到固定长度。"""
    out = np.zeros(length, dtype=np.float32)
    n = min(len(y), length)
    out[:n] = y[:n]
    return out


def compute_signal_power(x: np.ndarray) -> float:
    """计算信号均方功率。"""
    if x.size == 0:
        return 0.0
    return float(np.mean(x ** 2))


def add_white_noise_by_snr(
    clean: np.ndarray,
    snr_db: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按指定 SNR 添加白噪声。

    clean: [num_mics, num_samples]
    返回：
    - noisy: clean + noise
    - noise: 噪声矩阵
    """
    signal_power = compute_signal_power(clean)

    if signal_power < 1e-12:
        noise = np.zeros_like(clean, dtype=np.float32)
        return clean.astype(np.float32), noise

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=clean.shape
    ).astype(np.float32)

    noisy = clean + noise
    return noisy.astype(np.float32), noise.astype(np.float32)


def scale_scene_audio(
    source_images: np.ndarray,
    clean_mix: np.ndarray,
    noisy_mix: np.ndarray,
    noise: np.ndarray,
    target_peak: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    对同一场景的 source images、clean mix、noisy mix、noise 使用同一个缩放系数。

    这样可以保证：
    - noisy_mix == sum(source_images) + noise
    - clean_mix == sum(source_images)
    """
    peak = float(np.max(np.abs(noisy_mix))) if noisy_mix.size > 0 else 0.0

    if peak < 1e-8:
        return source_images, clean_mix, noisy_mix, noise, 1.0

    scale = min(1.0, target_peak / peak)

    return (
        (source_images * scale).astype(np.float32),
        (clean_mix * scale).astype(np.float32),
        (noisy_mix * scale).astype(np.float32),
        (noise * scale).astype(np.float32),
        float(scale)
    )


def write_multichannel_as_mono_files(
    signals: np.ndarray,
    out_dir: Path,
    prefix: str,
    sr: int = TARGET_SR
) -> List[str]:
    """
    将 [num_mics, num_samples] 的多麦克风信号保存为多个单通道 wav。

    返回相对项目根目录的路径列表。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    num_mics = signals.shape[0]

    for m in range(num_mics):
        out_path = out_dir / f"{prefix}_mic{m + 1:02d}.wav"
        sf.write(out_path, signals[m], sr, subtype=WAV_SUBTYPE)
        paths.append(rel_to_root(out_path))

    return paths


# =========================
# 场景配置生成
# =========================

def random_source_position(
    cage: Dict,
    rng: np.random.Generator
) -> List[float]:
    """在鸡笼内部随机生成一个合法声源坐标。"""
    length_x = float(cage["dimensions"]["length_x"])
    width_y = float(cage["dimensions"]["width_y"])
    height_z = float(cage["dimensions"]["height_z"])

    x_min = SOURCE_MARGIN_X
    x_max = length_x - SOURCE_MARGIN_X
    y_min = SOURCE_MARGIN_Y
    y_max = width_y - SOURCE_MARGIN_Y

    z_min = SOURCE_Z_RANGE[0]
    z_max = SOURCE_Z_RANGE[1]

    if z_max > height_z:
        raise ValueError(
            f"声源高度范围 {SOURCE_Z_RANGE} 超出鸡笼高度 {height_z}，请检查 SOURCE_Z_RANGE。"
        )

    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)

    if abs(z_max - z_min) < 1e-9:
        z = z_min
    else:
        z = rng.uniform(z_min, z_max)

    return [round(float(x), 3), round(float(y), 3), round(float(z), 3)]


def classify_scene_type(num_sources: int) -> str:
    """根据声源数量标记场景类型。"""
    if num_sources == 1:
        return "single_source"
    if num_sources == 2:
        return "dual_source"
    return "multi_source"


def planned_metrics_for_scene(num_sources: int) -> List[str]:
    """
    为场景预留后续评价指标字段。

    注意：
    - 当前脚本不计算定位和重建指标；
    - 后续 07/08/09/10 脚本会使用这些字段。
    """
    if num_sources == 1:
        return [
            "localization_error_2d_cm",
            "MAE_cm",
            "P90_cm",
            "Hit@10cm",
            "Hit@20cm",
            "miss_rate",
            "false_alarm_rate",
        ]

    return [
        "multi_source_localization_error_2d_cm",
        "source_count_error",
        "multi_source_hit_rate",
        "miss_rate",
        "false_alarm_rate",
    ]


def build_scene_configs(
    cage: Dict,
    mic_layouts: Dict,
    audio_files: List[Path]
) -> Dict:
    """
    构建第 1 周第 5 天的 20 个可复现仿真场景配置。

    该函数不是论文级每组100场景的正式实验矩阵。
    但它会为每个场景写入后续论文级实验所需的元信息：
    - experiment_id
    - group_id
    - task_type
    - scene_type
    - ground truth source position
    - planned_metrics
    - seed
    """
    layout_names = LAYOUT_CHOICES_FOR_DAY5

    for name in layout_names:
        if name not in mic_layouts["layouts"]:
            raise KeyError(f"mic_layouts.yaml 中缺少布局：{name}")
    scenes = []

    for scene_idx in range(NUM_SCENES):
        scene_seed = int(GLOBAL_SEED + scene_idx * 97)
        scene_rng = np.random.default_rng(scene_seed)

        scene_id = f"scene_{scene_idx + 1:03d}"

        layout_name = str(scene_rng.choice(layout_names))
        num_sources = int(scene_rng.choice(NUM_SOURCES_CHOICES))

        rt60 = float(scene_rng.uniform(RT60_RANGE_SEC[0], RT60_RANGE_SEC[1]))
        snr_db = float(scene_rng.uniform(SNR_RANGE_DB[0], SNR_RANGE_DB[1]))

        scene_type = classify_scene_type(num_sources)

        # 当前为功能验证，不归入正式 E1～E10 组
        experiment_id = "W1D5"
        group_id = f"W1D5_{scene_type}_{layout_name}"
        task_type = "simulation_function_check"

        sources = []

        chosen_indices = scene_rng.choice(
            len(audio_files),
            size=num_sources,
            replace=False if len(audio_files) >= num_sources else True
        )

        for s_idx, audio_index in enumerate(chosen_indices):
            audio_path = audio_files[int(audio_index)]

            delay_sec = float(scene_rng.uniform(
                SOURCE_DELAY_RANGE_SEC[0],
                SOURCE_DELAY_RANGE_SEC[1],
            ))

            gain = float(scene_rng.uniform(
                SOURCE_GAIN_RANGE[0],
                SOURCE_GAIN_RANGE[1],
            ))

            position = random_source_position(cage, scene_rng)

            sources.append({
                "source_id": f"S{s_idx + 1:02d}",
                "audio_path": rel_to_root(audio_path),

                # 三维真值坐标，后续定位和重建都会用
                "position": position,
                "x_gt": position[0],
                "y_gt": position[1],
                "z_gt": position[2],

                # 二维定位真值坐标，后续 MAE/P90/命中率使用
                "x_gt_2d": position[0],
                "y_gt_2d": position[1],

                "delay_sec": round(delay_sec, 3),
                "gain": round(gain, 3),
            })

        scenes.append({
            "scene_id": scene_id,
            "run_mode": RUN_MODE,
            "paper_ready_metadata": PAPER_READY_METADATA,

            # 后续论文级实验矩阵接口字段
            "experiment_id": experiment_id,
            "group_id": group_id,
            "task_type": task_type,
            "scene_type": scene_type,
            "planned_metrics": planned_metrics_for_scene(num_sources),

            # 可复现配置
            "seed": scene_seed,
            "global_seed": GLOBAL_SEED,

            # 声学仿真配置
            "fs": TARGET_SR,
            "duration_sec": SCENE_DURATION_SEC,
            "mic_layout": layout_name,
            "rt60_sec": round(rt60, 3),
            "snr_db": round(snr_db, 2),
            "num_sources": num_sources,

            # 声源真值
            "sources": sources,
        })

    return {
        "name": "week1_day5_simulation_scenes",
        "description": "第1周第5天仿真器功能验证场景，同时预留论文级实验元数据。",
        "run_mode": RUN_MODE,
        "paper_ready_metadata": PAPER_READY_METADATA,
        "global_seed": GLOBAL_SEED,
        "num_scenes": NUM_SCENES,
        "scene_duration_sec": SCENE_DURATION_SEC,
        "note": (
            "当前仅生成20个测试场景，用于验证仿真器功能。"
            "论文级 E1～E10 每组50/100场景将在后续脚本中生成。"
        ),
        "planned_next_experiments": PLANNED_NEXT_EXPERIMENTS,
        "scenes": scenes,
    }


# =========================
# Pyroomacoustics 仿真核心
# =========================

def get_room_dimensions(cage: Dict) -> List[float]:
    """从 cage.yaml 中读取房间尺寸。"""
    return [
        float(cage["dimensions"]["length_x"]),
        float(cage["dimensions"]["width_y"]),
        float(cage["dimensions"]["height_z"]),
    ]


def get_mic_positions(mic_layouts: Dict, layout_name: str) -> np.ndarray:
    """
    获取麦克风坐标矩阵。

    返回形状：[3, num_mics]
    Pyroomacoustics 的 MicrophoneArray 需要该格式。
    """
    microphones = mic_layouts["layouts"][layout_name]["microphones"]
    positions = [m["position"] for m in microphones]
    return np.array(positions, dtype=np.float64).T


def build_room_with_sources(
    cage: Dict,
    mic_layouts: Dict,
    scene: Dict
):
    """
    构建 Pyroomacoustics 房间，并计算 RIR。

    注意：
    - 声源 signal 不直接交给 room.simulate；
    - 本脚本使用 room.rir 自己卷积生成每个 source image；
    - 这样便于严格验证 mixed == sum(source_images) + noise。
    """
    room_dim = get_room_dimensions(cage)
    fs = int(scene["fs"])
    rt60 = float(scene["rt60_sec"])

    # 根据目标 RT60 反推吸声系数和 image-source 最大阶数
    absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(absorption),
        max_order=max_order,
        air_absorption=True,
    )

    mic_positions = get_mic_positions(mic_layouts, scene["mic_layout"])
    mic_array = pra.MicrophoneArray(mic_positions, fs=fs)
    room.add_microphone_array(mic_array)

    for src in scene["sources"]:
        room.add_source(src["position"])

    room.compute_rir()

    return room, absorption, max_order


def save_rirs(room, scene_dir: Path):
    """保存 RIR 文件。"""
    rir_dir = scene_dir / "rir"
    rir_dir.mkdir(parents=True, exist_ok=True)

    rir_dict = {}

    num_mics = len(room.rir)
    num_sources = len(room.rir[0]) if num_mics > 0 else 0

    for m in range(num_mics):
        for s in range(num_sources):
            key = f"rir_s{s + 1:02d}_mic{m + 1:02d}"
            rir_dict[key] = np.asarray(room.rir[m][s], dtype=np.float32)

    rir_path = rir_dir / "rirs.npz"
    np.savez_compressed(rir_path, **rir_dict)

    return rel_to_root(rir_path)


def save_image_source_geometry(room, scene_dir: Path):
    """
    保存 image-source 模型中的镜像声源几何位置。

    这不是音频，而是 Pyroomacoustics image-source model 生成的镜像声源坐标。
    """
    image_dir = scene_dir / "image_source_geometry"
    image_dir.mkdir(parents=True, exist_ok=True)

    data = {}

    for s_idx, src in enumerate(room.sources):
        if hasattr(src, "images"):
            images = np.asarray(src.images, dtype=np.float32)
            data[f"source_{s_idx + 1:02d}_images"] = images

        if hasattr(src, "damping"):
            damping = np.asarray(src.damping, dtype=np.float32)
            data[f"source_{s_idx + 1:02d}_damping"] = damping

        if hasattr(src, "orders"):
            orders = np.asarray(src.orders, dtype=np.int32)
            data[f"source_{s_idx + 1:02d}_orders"] = orders

    image_path = image_dir / "image_source_geometry.npz"
    np.savez_compressed(image_path, **data)

    return rel_to_root(image_path)


def convolve_sources_with_rir(
    room,
    scene: Dict,
    scene_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, List[Dict]]:
    """
    使用 RIR 将每个源卷积到每个麦克风。

    返回：
    - source_images: [num_sources, num_mics, num_samples]
    - clean_mix: [num_mics, num_samples]
    - noisy_mix: [num_mics, num_samples]
    - noise: [num_mics, num_samples]
    - scene_info: 场景附加信息
    - source_records: 声源记录
    """
    fs = int(scene["fs"])
    duration_sec = float(scene["duration_sec"])
    num_samples = int(duration_sec * fs)

    num_sources = len(scene["sources"])
    num_mics = len(room.rir)

    source_images = np.zeros(
        (num_sources, num_mics, num_samples),
        dtype=np.float32
    )

    source_records = []

    for s_idx, src in enumerate(scene["sources"]):
        audio_path = ROOT / src["audio_path"]
        y = load_audio_mono(audio_path, target_sr=fs)
        y = normalize_peak(y, target_peak=0.80)

        delay_samples = int(round(float(src["delay_sec"]) * fs))
        gain = float(src["gain"])

        delayed = np.zeros(num_samples, dtype=np.float32)

        if delay_samples < num_samples:
            available = num_samples - delay_samples
            n = min(len(y), available)
            delayed[delay_samples:delay_samples + n] = y[:n] * gain

        for m_idx in range(num_mics):
            rir = np.asarray(room.rir[m_idx][s_idx], dtype=np.float32)
            conv = fftconvolve(delayed, rir, mode="full")
            conv = pad_or_crop(conv.astype(np.float32), num_samples)
            source_images[s_idx, m_idx, :] = conv

        source_records.append({
            "scene_id": scene["scene_id"],
            "run_mode": scene["run_mode"],
            "experiment_id": scene["experiment_id"],
            "group_id": scene["group_id"],
            "task_type": scene["task_type"],
            "scene_type": scene["scene_type"],
            "source_id": src["source_id"],
            "audio_path": src["audio_path"],

            # 真值坐标，后续定位评价使用
            "x_gt": src["x_gt"],
            "y_gt": src["y_gt"],
            "z_gt": src["z_gt"],
            "x_gt_2d": src["x_gt_2d"],
            "y_gt_2d": src["y_gt_2d"],

            "delay_sec": src["delay_sec"],
            "gain": src["gain"],
        })

    clean_mix = np.sum(source_images, axis=0)

    rng = np.random.default_rng(int(scene["seed"]))
    noisy_mix, noise = add_white_noise_by_snr(
        clean=clean_mix,
        snr_db=float(scene["snr_db"]),
        rng=rng,
    )

    source_images, clean_mix, noisy_mix, noise, scale = scale_scene_audio(
        source_images=source_images,
        clean_mix=clean_mix,
        noisy_mix=noisy_mix,
        noise=noise,
        target_peak=0.95,
    )

    scene_info = {
        "scale": scale,
        "num_sources": num_sources,
        "num_mics": num_mics,
        "num_samples": num_samples,
    }

    return source_images, clean_mix, noisy_mix, noise, scene_info, source_records


def save_scene_audio(
    scene: Dict,
    scene_dir: Path,
    source_images: np.ndarray,
    clean_mix: np.ndarray,
    noisy_mix: np.ndarray,
    noise: np.ndarray,
):
    """保存场景音频，包括 source images、clean mix、noisy mix 和 noise。"""
    fs = int(scene["fs"])

    source_image_dir = scene_dir / "source_images"
    clean_mix_dir = scene_dir / "mixed_clean"
    noisy_mix_dir = scene_dir / "mixed_noisy"
    noise_dir = scene_dir / "noise"

    source_paths = []

    for s_idx in range(source_images.shape[0]):
        src_dir = source_image_dir / f"source_{s_idx + 1:02d}"
        paths = write_multichannel_as_mono_files(
            source_images[s_idx],
            src_dir,
            prefix=f"source_{s_idx + 1:02d}",
            sr=fs,
        )
        source_paths.extend(paths)

    clean_paths = write_multichannel_as_mono_files(
        clean_mix,
        clean_mix_dir,
        prefix="mixed_clean",
        sr=fs,
    )

    noisy_paths = write_multichannel_as_mono_files(
        noisy_mix,
        noisy_mix_dir,
        prefix="mixed_noisy",
        sr=fs,
    )

    noise_paths = write_multichannel_as_mono_files(
        noise,
        noise_dir,
        prefix="noise",
        sr=fs,
    )

    return {
        "source_image_paths": source_paths,
        "clean_mix_paths": clean_paths,
        "noisy_mix_paths": noisy_paths,
        "noise_paths": noise_paths,
    }


def validate_scene_sum(
    source_images: np.ndarray,
    clean_mix: np.ndarray,
    noisy_mix: np.ndarray,
    noise: np.ndarray,
) -> Dict:
    """
    验证混合信号与各 source image 之和一致。

    检查：
    1. clean_mix == sum(source_images)
    2. noisy_mix == sum(source_images) + noise
    """
    source_sum = np.sum(source_images, axis=0)

    clean_err = float(np.max(np.abs(clean_mix - source_sum)))
    noisy_err = float(np.max(np.abs(noisy_mix - source_sum - noise)))

    return {
        "clean_sum_max_abs_error": clean_err,
        "noisy_sum_max_abs_error": noisy_err,
        "clean_sum_pass": clean_err < 1e-5,
        "noisy_sum_pass": noisy_err < 1e-5,
    }


def simulate_one_scene(
    cage: Dict,
    mic_layouts: Dict,
    scene: Dict
) -> Tuple[Dict, List[Dict], Dict]:
    """仿真单个场景。"""
    scene_id = scene["scene_id"]
    scene_dir = SIM_DATA_ROOT / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    room, absorption, max_order = build_room_with_sources(
        cage=cage,
        mic_layouts=mic_layouts,
        scene=scene,
    )

    rir_path = save_rirs(room, scene_dir)
    image_source_geometry_path = save_image_source_geometry(room, scene_dir)

    source_images, clean_mix, noisy_mix, noise, scene_info, source_records = (
        convolve_sources_with_rir(room, scene, scene_dir)
    )

    audio_paths = save_scene_audio(
        scene=scene,
        scene_dir=scene_dir,
        source_images=source_images,
        clean_mix=clean_mix,
        noisy_mix=noisy_mix,
        noise=noise,
    )

    validation = validate_scene_sum(
        source_images=source_images,
        clean_mix=clean_mix,
        noisy_mix=noisy_mix,
        noise=noise,
    )

    scene_record = {
        "scene_id": scene_id,

        # 论文级衔接字段
        "run_mode": scene["run_mode"],
        "paper_ready_metadata": scene["paper_ready_metadata"],
        "experiment_id": scene["experiment_id"],
        "group_id": scene["group_id"],
        "task_type": scene["task_type"],
        "scene_type": scene["scene_type"],
        "planned_metrics": ";".join(scene["planned_metrics"]),

        # 可复现字段
        "seed": scene["seed"],
        "global_seed": scene["global_seed"],

        # 场景参数
        "mic_layout": scene["mic_layout"],
        "num_sources": scene_info["num_sources"],
        "num_mics": scene_info["num_mics"],
        "duration_sec": scene["duration_sec"],
        "fs": scene["fs"],
        "rt60_sec": scene["rt60_sec"],
        "snr_db": scene["snr_db"],
        "absorption": float(absorption),
        "max_order": int(max_order),
        "scale": scene_info["scale"],

        # 路径
        "scene_dir": rel_to_root(scene_dir),
        "rir_path": rir_path,
        "image_source_geometry_path": image_source_geometry_path,
        "first_mixed_clean_path": audio_paths["clean_mix_paths"][0],
        "first_mixed_noisy_path": audio_paths["noisy_mix_paths"][0],

        # 求和验证
        "clean_sum_max_abs_error": validation["clean_sum_max_abs_error"],
        "noisy_sum_max_abs_error": validation["noisy_sum_max_abs_error"],
        "clean_sum_pass": validation["clean_sum_pass"],
        "noisy_sum_pass": validation["noisy_sum_pass"],
    }

    validation_record = {
        "scene_id": scene_id,
        "run_mode": scene["run_mode"],
        "experiment_id": scene["experiment_id"],
        "group_id": scene["group_id"],
        "scene_type": scene["scene_type"],
        **validation,
    }

    return scene_record, source_records, validation_record


# =========================
# 报告生成
# =========================

def write_quality_report(
    scene_df: pd.DataFrame,
    source_df: pd.DataFrame,
    validation_df: pd.DataFrame
):
    """生成 Markdown 质量报告。"""
    total_scenes = len(scene_df)
    single_source_scenes = int((scene_df["num_sources"] == 1).sum()) if total_scenes > 0 else 0
    dual_source_scenes = int((scene_df["num_sources"] == 2).sum()) if total_scenes > 0 else 0
    multi_source_scenes = int((scene_df["num_sources"] >= 2).sum()) if total_scenes > 0 else 0

    clean_pass = int(validation_df["clean_sum_pass"].sum()) if len(validation_df) > 0 else 0
    noisy_pass = int(validation_df["noisy_sum_pass"].sum()) if len(validation_df) > 0 else 0

    max_clean_err = (
        float(validation_df["clean_sum_max_abs_error"].max())
        if len(validation_df) > 0 else np.nan
    )

    max_noisy_err = (
        float(validation_df["noisy_sum_max_abs_error"].max())
        if len(validation_df) > 0 else np.nan
    )

    pass_all = (
        total_scenes == NUM_SCENES
        and clean_pass == total_scenes
        and noisy_pass == total_scenes
    )

    lines = []
    lines.append("# 第 1 周第 5 天：单源和多源仿真器质量报告")
    lines.append("")
    lines.append("## 1. 当前任务定位")
    lines.append("")
    lines.append("本阶段属于第 1 周仿真器功能验证，不是论文级 E1～E10 正式实验。")
    lines.append("当前生成 20 个测试场景，用于验证 RIR、source image、mixed audio、随机种子复现和求和一致性。")
    lines.append("同时，本脚本已经在 YAML 和 CSV 中写入后续论文级实验所需的元数据字段。")
    lines.append("")

    lines.append("## 2. 功能目标")
    lines.append("")
    lines.append("- 使用 Pyroomacoustics 实现 RIR 生成")
    lines.append("- 保存 image-source 模型几何信息")
    lines.append("- 生成单源与多源混合音频")
    lines.append("- 支持声源起始时间、增益、RT60、SNR 和随机种子")
    lines.append("- 生成 20 个可复现测试场景")
    lines.append("- 验证 clean mix 与 source image 求和一致")
    lines.append("- 验证 noisy mix 与 source image + noise 求和一致")
    lines.append("- 预留 experiment_id、group_id、ground truth 坐标和 planned_metrics 字段")
    lines.append("")

    lines.append("## 3. 总体统计")
    lines.append("")
    lines.append(f"- 场景数量：`{total_scenes}`")
    lines.append(f"- 单源场景数量：`{single_source_scenes}`")
    lines.append(f"- 双源场景数量：`{dual_source_scenes}`")
    lines.append(f"- 多源场景数量：`{multi_source_scenes}`")
    lines.append(f"- 声源总数：`{len(source_df)}`")
    lines.append(f"- clean mix 求和验证通过场景数：`{clean_pass}`")
    lines.append(f"- noisy mix 求和验证通过场景数：`{noisy_pass}`")
    lines.append(f"- clean mix 最大误差：`{max_clean_err:.8e}`")
    lines.append(f"- noisy mix 最大误差：`{max_noisy_err:.8e}`")
    lines.append("")

    if pass_all:
        lines.append("结论：今日验收通过，20 个场景均可复现，混合信号与 source image 求和关系一致。")
    else:
        lines.append("结论：今日验收未完全通过，需要检查 validation CSV。")

    lines.append("")
    lines.append("## 4. 后续论文级实验衔接")
    lines.append(
        "- 后续论文级实验将完整覆盖 E1～E10，并输出 localization_results.csv、separation_results.csv、experiment_summary.csv 和 daily_log.csv。")
    lines.append("")
    lines.append("当前 20 个场景不作为论文级统计实验结果。后续正式实验将由 `06_build_paper_sim_scenes.py` 扩展。")
    lines.append("")
    lines.append("后续建议：")
    lines.append("")
    lines.append("- E1：麦克风数量 4/6/8/12，每组 100 场景，计算 MAE、P90、10 cm 命中率")
    lines.append("- E3：SNR 0/5/10/20/30 dB，每组 100 场景，计算误差和漏检率")
    lines.append("- E5：RT60 0.1/0.3/0.5/0.7 s，每组 100 场景，计算误差和峰值比")
    lines.append("- E6：单通道 / DAS / MVDR / LCMV，每组 100 场景，计算 SI-SDRi 和 Log-Mel 距离")
    lines.append("")

    lines.append("## 5. 输出文件")
    lines.append("")
    lines.append(f"- 场景配置：`{rel_to_root(SCENE_CONFIG_YAML)}`")
    lines.append(f"- 场景清单：`{rel_to_root(SCENE_MANIFEST_CSV)}`")
    lines.append(f"- 声源清单：`{rel_to_root(SOURCE_MANIFEST_CSV)}`")
    lines.append(f"- 验证结果：`{rel_to_root(VALIDATION_CSV)}`")
    lines.append(f"- 仿真数据目录：`{rel_to_root(SIM_DATA_ROOT)}`")
    lines.append("")

    QUALITY_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# =========================
# 主函数
# =========================

def main():
    """主函数。"""
    warnings.filterwarnings("ignore")

    if CLEAR_OLD_OUTPUTS:
        reset_output_dirs()
    else:
        ensure_output_dirs()

    print("========== 第 1 周第 5 天：单源和多源仿真器 ==========")
    print(f"[INFO] 项目根目录: {ROOT}")
    print(f"[INFO] cage 配置: {CAGE_YAML}")
    print(f"[INFO] mic layouts 配置: {MIC_LAYOUTS_YAML}")
    print(f"[INFO] 输入片段目录: {INPUT_SEGMENT_ROOT}")
    print(f"[INFO] 输出仿真目录: {SIM_DATA_ROOT}")
    print(f"[INFO] 当前运行模式: {RUN_MODE}")
    print("")

    if not CAGE_YAML.exists():
        raise FileNotFoundError(f"缺少 cage.yaml: {CAGE_YAML}")

    if not MIC_LAYOUTS_YAML.exists():
        raise FileNotFoundError(f"缺少 mic_layouts.yaml: {MIC_LAYOUTS_YAML}")

    cage = load_yaml(CAGE_YAML)
    mic_layouts = load_yaml(MIC_LAYOUTS_YAML)

    audio_files = find_audio_segments()
    print(f"[INFO] 找到可用音频片段: {len(audio_files)}")

    if len(audio_files) == 0:
        raise FileNotFoundError(
            f"未在 {INPUT_SEGMENT_ROOT} 下找到音频片段，请先完成第 1 周第 3 天预处理。"
        )

    scene_config = build_scene_configs(
        cage=cage,
        mic_layouts=mic_layouts,
        audio_files=audio_files,
    )

    save_yaml(scene_config, SCENE_CONFIG_YAML)
    print(f"[OK] 已生成场景配置: {SCENE_CONFIG_YAML}")

    scene_records = []
    source_records_all = []
    validation_records = []

    for scene in scene_config["scenes"]:
        print(f"[RUN] 正在仿真 {scene['scene_id']} | "
              f"type={scene['scene_type']} | "
              f"layout={scene['mic_layout']} | "
              f"sources={len(scene['sources'])} | "
              f"RT60={scene['rt60_sec']} | "
              f"SNR={scene['snr_db']}")

        scene_record, source_records, validation_record = simulate_one_scene(
            cage=cage,
            mic_layouts=mic_layouts,
            scene=scene,
        )

        scene_records.append(scene_record)
        source_records_all.extend(source_records)
        validation_records.append(validation_record)

    scene_df = pd.DataFrame(scene_records)
    source_df = pd.DataFrame(source_records_all)
    validation_df = pd.DataFrame(validation_records)

    scene_df.to_csv(SCENE_MANIFEST_CSV, index=False, encoding="utf-8-sig")
    source_df.to_csv(SOURCE_MANIFEST_CSV, index=False, encoding="utf-8-sig")
    validation_df.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")

    write_quality_report(scene_df, source_df, validation_df)

    total_scenes = len(scene_df)
    clean_pass = int(validation_df["clean_sum_pass"].sum())
    noisy_pass = int(validation_df["noisy_sum_pass"].sum())

    print("")
    print("========== 处理完成 ==========")
    print(f"场景数量: {total_scenes}")
    print(f"clean mix 求和验证通过: {clean_pass}/{total_scenes}")
    print(f"noisy mix 求和验证通过: {noisy_pass}/{total_scenes}")
    print(f"场景配置: {SCENE_CONFIG_YAML}")
    print(f"场景清单: {SCENE_MANIFEST_CSV}")
    print(f"声源清单: {SOURCE_MANIFEST_CSV}")
    print(f"验证结果: {VALIDATION_CSV}")
    print(f"质量报告: {QUALITY_REPORT_MD}")
    print(f"仿真数据目录: {SIM_DATA_ROOT}")

    if total_scenes == NUM_SCENES and clean_pass == total_scenes and noisy_pass == total_scenes:
        print("[PASS] 今日验收通过：20 场景可复现，混合信号与 source image 求和一致。")
        print("[INFO] 已预留论文级实验元数据，可供后续 06～10 脚本继续使用。")
    else:
        print("[WARN] 今日验收未完全通过，请检查 validation CSV。")


if __name__ == "__main__":
    main()