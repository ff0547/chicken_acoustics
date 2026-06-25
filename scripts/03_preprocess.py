# -*- coding: utf-8 -*-
"""
第 1 周周三：音频清洗与标准化

功能：
1. 扫描 data/raw_sources 下的正式音频数据
2. 转单声道
3. 重采样到 48 kHz
4. 去直流
5. 静音裁剪
6. 峰值归一化
7. 输出清洗后的完整音频到 data/processed_sources
8. 将音频切分为 1～5 s 的有效片段
9. 输出切分片段到 data/processed_segments
10. 绘制波形、频谱、Log-Mel 样例图
11. 生成质量报告

注意：
- 不处理 data/candidate_sources，因为 ChickenLanguageDataset 许可证不明确
- 输出统一为 .wav
- 代码注释使用中文
"""

from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm


# =========================
# 基本配置
# =========================

ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = ROOT / "data" / "raw_sources"
PROCESSED_ROOT = ROOT / "data" / "processed_sources"
SEGMENT_ROOT = ROOT / "data" / "processed_segments"

REPORT_DIR = ROOT / "results" / "week1" / "day3" / "preprocess"
EXAMPLE_DIR = REPORT_DIR / "examples"

FILE_REPORT_CSV = REPORT_DIR / "preprocess_file_report.csv"
SEGMENT_REPORT_CSV = REPORT_DIR / "segment_manifest.csv"
QUALITY_REPORT_MD = REPORT_DIR / "preprocess_quality_report.md"

TARGET_SR = 48000
TARGET_PEAK = 0.95

# 静音裁剪阈值，数值越小裁剪越激进
TRIM_TOP_DB = 35

# 切分片段要求：1～5 秒
MIN_SEGMENT_SEC = 1.0
MAX_SEGMENT_SEC = 5.0

MIN_SEGMENT_SAMPLES = int(MIN_SEGMENT_SEC * TARGET_SR)
MAX_SEGMENT_SAMPLES = int(MAX_SEGMENT_SEC * TARGET_SR)

# 生成样例图的片段数量
NUM_EXAMPLE_SEGMENTS = 3

# 是否清空旧输出，避免上次运行残留文件影响统计
CLEAR_OLD_OUTPUTS = True

AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
}


# =========================
# 通用工具函数
# =========================

def reset_output_dirs():
    """清空旧的输出目录，避免重复运行时残留旧文件。"""
    targets = [
        PROCESSED_ROOT,
        SEGMENT_ROOT,
        REPORT_DIR,
    ]

    for path in targets:
        if path.exists():
            shutil.rmtree(path)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    SEGMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs():
    """创建输出目录。"""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    SEGMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def find_audio_files(input_root: Path):
    """递归查找正式数据目录下的音频文件。"""
    if not input_root.exists():
        return []

    audio_files = []
    for p in input_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            audio_files.append(p)

    return sorted(audio_files)


def safe_float(x):
    """把数值安全转为普通 float，便于写入 CSV。"""
    try:
        return float(x)
    except Exception:
        return np.nan


def rel_to_root(path: Path):
    """生成相对于项目根目录的路径字符串。"""
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def get_dataset_name(src: Path):
    """
    根据 raw_sources 下的第一级目录判断数据集名称。

    例：
    data/raw_sources/mendeley_poultry_vocalization/Healthy/a.wav
    dataset = mendeley_poultry_vocalization
    """
    try:
        rel_parts = src.relative_to(INPUT_ROOT).parts
        if len(rel_parts) > 0:
            return rel_parts[0]
    except Exception:
        pass
    return "unknown"


# =========================
# 音频处理函数
# =========================

def load_audio(path: Path):
    """
    读取音频。

    librosa 会自动将整数 PCM 转为 float32。
    mono=False 是为了先记录原始通道数，再手动转单声道。
    """
    y, sr = librosa.load(path, sr=None, mono=False)

    if y.ndim == 1:
        channels = 1
        y_mono = y
    else:
        # librosa 读取多声道时通常为 [通道数, 采样点数]
        channels = y.shape[0]
        y_mono = np.mean(y, axis=0)

    return y_mono.astype(np.float32), int(sr), int(channels)


def remove_dc(y: np.ndarray):
    """去除直流分量，即减去均值。"""
    if len(y) == 0:
        return y
    return (y - np.mean(y)).astype(np.float32)


def resample_to_target(y: np.ndarray, sr: int):
    """重采样到目标采样率。"""
    if sr == TARGET_SR:
        return y.astype(np.float32), TARGET_SR

    y2 = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y2.astype(np.float32), TARGET_SR


def trim_silence(y: np.ndarray):
    """裁剪首尾静音。"""
    if len(y) == 0:
        return y, 0, 0

    y_trim, index = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    start, end = int(index[0]), int(index[1])
    return y_trim.astype(np.float32), start, end


def peak_normalize(y: np.ndarray):
    """峰值归一化，目标峰值为 TARGET_PEAK。"""
    if len(y) == 0:
        return y, 0.0, 0.0

    peak_before = np.max(np.abs(y))

    if peak_before < 1e-8:
        return y.astype(np.float32), safe_float(peak_before), safe_float(peak_before)

    y_norm = y / peak_before * TARGET_PEAK
    peak_after = np.max(np.abs(y_norm))

    return y_norm.astype(np.float32), safe_float(peak_before), safe_float(peak_after)


def rms(y: np.ndarray):
    """计算 RMS 能量。"""
    if len(y) == 0:
        return 0.0
    return safe_float(np.sqrt(np.mean(y ** 2)))


def count_clipped_samples(y: np.ndarray):
    """
    统计削波样本数。

    这里按 |x| >= 0.999 判断。
    正常峰值归一化到 0.95 后，削波样本应为 0。
    """
    if len(y) == 0:
        return 0
    return int(np.sum(np.abs(y) >= 0.999))


def make_processed_path(src: Path):
    """
    根据输入路径生成清洗后完整音频输出路径。

    例：
    data/raw_sources/A/B/a.mp3
    -> data/processed_sources/A/B/a.wav
    """
    rel = src.relative_to(INPUT_ROOT)
    out = PROCESSED_ROOT / rel.with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def make_segment_path(processed_path: Path, segment_index: int):
    """
    根据清洗后完整音频路径生成切分片段路径。

    例：
    data/processed_sources/A/B/a.wav
    -> data/processed_segments/A/B/a_seg000.wav
    """
    rel = processed_path.relative_to(PROCESSED_ROOT)
    rel_no_suffix = rel.with_suffix("")

    out_dir = SEGMENT_ROOT / rel_no_suffix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{rel_no_suffix.name}_seg{segment_index:03d}.wav"
    return out_dir / out_name


def split_into_segments(y: np.ndarray):
    """
    将音频切分为 1～5 秒片段。

    规则：
    - 每段最长 5 秒；
    - 最后一段如果不足 1 秒，则丢弃；
    - 所有保留片段均满足 1～5 秒。
    """
    segments = []

    if len(y) < MIN_SEGMENT_SAMPLES:
        return segments

    start = 0
    segment_index = 0

    while start < len(y):
        end = min(start + MAX_SEGMENT_SAMPLES, len(y))
        seg = y[start:end]

        if len(seg) >= MIN_SEGMENT_SAMPLES:
            segments.append({
                "segment_index": segment_index,
                "start_sample": int(start),
                "end_sample": int(end),
                "audio": seg.astype(np.float32),
            })
            segment_index += 1

        start = end

    return segments


# =========================
# 可视化函数
# =========================

def plot_waveform(y: np.ndarray, sr: int, out_path: Path, title: str):
    """绘制波形图。"""
    duration = len(y) / sr
    t = np.linspace(0, duration, num=len(y), endpoint=False)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spectrum(y: np.ndarray, sr: int, out_path: Path, title: str):
    """绘制频谱图。"""
    if len(y) == 0:
        return

    # 加窗后做快速傅里叶变换
    window = np.hanning(len(y))
    y_win = y * window

    spec = np.fft.rfft(y_win)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(y_win), d=1.0 / sr)

    # 归一化为 dB，便于观察
    mag_db = 20 * np.log10(mag / (np.max(mag) + 1e-12) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag_db, linewidth=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.xlim(0, min(sr / 2, 12000))
    plt.ylim(-100, 5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_logmel(y: np.ndarray, sr: int, out_path: Path, title: str):
    """绘制 Log-Mel 频谱图。"""
    if len(y) == 0:
        return

    n_fft = 2048
    hop_length = 512
    n_mels = 128

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_example_plots(segment_records):
    """从有效片段中选择若干个样例，绘制波形、频谱和 Log-Mel 图。"""
    valid_segments = [
        r for r in segment_records
        if r.get("is_valid", False) and r.get("segment_path", "")
    ]

    if len(valid_segments) == 0:
        print("[WARN] 没有有效片段，无法生成样例图。")
        return []

    selected = valid_segments[:NUM_EXAMPLE_SEGMENTS]
    example_outputs = []

    for i, record in enumerate(selected, start=1):
        segment_path = ROOT / record["segment_path"]

        y, sr = librosa.load(segment_path, sr=None, mono=True)
        y = y.astype(np.float32)

        prefix = f"sample_{i:02d}"

        waveform_path = EXAMPLE_DIR / f"{prefix}_waveform.png"
        spectrum_path = EXAMPLE_DIR / f"{prefix}_spectrum.png"
        logmel_path = EXAMPLE_DIR / f"{prefix}_logmel.png"

        title_base = f"{prefix}: {segment_path.name}"

        plot_waveform(
            y,
            sr,
            waveform_path,
            title=f"Waveform - {title_base}",
        )

        plot_spectrum(
            y,
            sr,
            spectrum_path,
            title=f"Spectrum - {title_base}",
        )

        plot_logmel(
            y,
            sr,
            logmel_path,
            title=f"Log-Mel Spectrogram - {title_base}",
        )

        example_outputs.extend([
            rel_to_root(waveform_path),
            rel_to_root(spectrum_path),
            rel_to_root(logmel_path),
        ])

    return example_outputs


# =========================
# 单文件处理函数
# =========================

def process_one_file(src: Path):
    """处理单个原始音频文件，返回文件记录和片段记录。"""
    file_record = {
        "source_path": rel_to_root(src),
        "processed_path": "",
        "dataset": get_dataset_name(src),
        "original_sr": np.nan,
        "target_sr": TARGET_SR,
        "original_channels": np.nan,
        "original_duration_sec": np.nan,
        "processed_duration_sec": np.nan,
        "trim_start_sample": np.nan,
        "trim_end_sample": np.nan,
        "peak_before_norm": np.nan,
        "peak_after_norm": np.nan,
        "rms_after": np.nan,
        "file_clipped_samples": np.nan,
        "num_segments": 0,
        "is_processed": False,
        "reason": "",
    }

    segment_records = []

    try:
        # 1. 读取音频并转单声道
        y, sr, channels = load_audio(src)

        file_record["original_sr"] = int(sr)
        file_record["original_channels"] = int(channels)
        file_record["original_duration_sec"] = safe_float(len(y) / sr)

        if len(y) == 0:
            file_record["reason"] = "empty_audio"
            return file_record, segment_records

        # 2. 去直流
        y = remove_dc(y)

        # 3. 重采样到 48 kHz
        y, sr2 = resample_to_target(y, sr)

        # 4. 静音裁剪
        y, trim_start, trim_end = trim_silence(y)

        file_record["trim_start_sample"] = int(trim_start)
        file_record["trim_end_sample"] = int(trim_end)

        if len(y) == 0:
            file_record["reason"] = "all_silence_after_trim"
            return file_record, segment_records

        # 5. 裁剪后再次去直流
        y = remove_dc(y)

        # 6. 峰值归一化
        y, peak_before, peak_after = peak_normalize(y)

        # 7. 防止极端浮点误差
        y = np.clip(y, -0.999, 0.999).astype(np.float32)

        processed_duration = len(y) / TARGET_SR
        file_clipped = count_clipped_samples(y)

        file_record["processed_duration_sec"] = safe_float(processed_duration)
        file_record["peak_before_norm"] = safe_float(peak_before)
        file_record["peak_after_norm"] = safe_float(peak_after)
        file_record["rms_after"] = rms(y)
        file_record["file_clipped_samples"] = int(file_clipped)

        if file_clipped > 0:
            file_record["reason"] = "clipping_detected_after_normalization"
            return file_record, segment_records

        # 8. 保存清洗后的完整音频
        processed_path = make_processed_path(src)
        sf.write(processed_path, y, TARGET_SR, subtype="PCM_16")

        file_record["processed_path"] = rel_to_root(processed_path)
        file_record["is_processed"] = True

        # 9. 切分为 1～5 秒片段
        segments = split_into_segments(y)

        if len(segments) == 0:
            file_record["reason"] = "no_valid_segment_1_to_5_sec"
            return file_record, segment_records

        for seg_info in segments:
            segment_index = seg_info["segment_index"]
            seg_audio = seg_info["audio"]

            segment_path = make_segment_path(processed_path, segment_index)
            sf.write(segment_path, seg_audio, TARGET_SR, subtype="PCM_16")

            seg_duration = len(seg_audio) / TARGET_SR
            seg_peak = np.max(np.abs(seg_audio)) if len(seg_audio) > 0 else 0.0
            seg_clipped = count_clipped_samples(seg_audio)

            segment_record = {
                "segment_path": rel_to_root(segment_path),
                "source_path": rel_to_root(src),
                "processed_path": rel_to_root(processed_path),
                "dataset": get_dataset_name(src),
                "segment_index": int(segment_index),
                "start_sample": int(seg_info["start_sample"]),
                "end_sample": int(seg_info["end_sample"]),
                "sr": TARGET_SR,
                "duration_sec": safe_float(seg_duration),
                "num_samples": int(len(seg_audio)),
                "peak": safe_float(seg_peak),
                "rms": rms(seg_audio),
                "clipped_samples": int(seg_clipped),
                "is_valid": bool(
                    MIN_SEGMENT_SEC <= seg_duration <= MAX_SEGMENT_SEC
                    and seg_clipped == 0
                ),
            }

            segment_records.append(segment_record)

        file_record["num_segments"] = int(len(segment_records))
        file_record["reason"] = "ok"

        return file_record, segment_records

    except Exception as e:
        file_record["reason"] = f"error: {type(e).__name__}: {e}"
        return file_record, segment_records


# =========================
# 报告生成函数
# =========================

def simple_markdown_table(headers, rows):
    """生成简单 Markdown 表格，避免依赖 tabulate。"""
    if len(headers) == 0:
        return ""

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        row_text = [str(x) for x in row]
        lines.append("| " + " | ".join(row_text) + " |")

    return "\n".join(lines)


def build_dataset_summary(file_df: pd.DataFrame, segment_df: pd.DataFrame):
    """按数据集统计文件数量和片段数量。"""
    datasets = sorted(set(file_df["dataset"].tolist())) if len(file_df) > 0 else []

    rows = []

    for ds in datasets:
        fsub = file_df[file_df["dataset"] == ds]
        ssub = segment_df[segment_df["dataset"] == ds] if len(segment_df) > 0 else pd.DataFrame()

        total_files = len(fsub)
        processed_files = int(fsub["is_processed"].sum()) if total_files > 0 else 0
        valid_segments = int(ssub["is_valid"].sum()) if len(ssub) > 0 else 0

        rows.append([
            ds,
            total_files,
            processed_files,
            valid_segments,
        ])

    return rows


def build_reason_summary(file_df: pd.DataFrame):
    """统计文件处理原因。"""
    if len(file_df) == 0:
        return []

    reason_counts = file_df["reason"].value_counts()
    rows = []

    for reason, count in reason_counts.items():
        rows.append([reason, int(count)])

    return rows


def write_quality_report(file_df: pd.DataFrame, segment_df: pd.DataFrame, example_outputs):
    """生成 Markdown 质量报告。"""
    total_files = len(file_df)
    processed_files = int(file_df["is_processed"].sum()) if total_files > 0 else 0

    total_segments = len(segment_df)
    valid_segments = int(segment_df["is_valid"].sum()) if total_segments > 0 else 0

    file_clipped_total = (
        int(file_df["file_clipped_samples"].fillna(0).sum())
        if total_files > 0 else 0
    )

    segment_clipped_total = (
        int(segment_df["clipped_samples"].fillna(0).sum())
        if total_segments > 0 else 0
    )

    pass_check = (
        valid_segments >= 80
        and file_clipped_total == 0
        and segment_clipped_total == 0
    )

    dataset_rows = build_dataset_summary(file_df, segment_df)
    reason_rows = build_reason_summary(file_df)

    lines = []

    lines.append("# 第 1 周周三：音频清洗与标准化质量报告")
    lines.append("")
    lines.append("## 1. 处理目标")
    lines.append("")
    lines.append("- 输入目录：`data/raw_sources`")
    lines.append("- 清洗音频输出目录：`data/processed_sources`")
    lines.append("- 切分片段输出目录：`data/processed_segments`")
    lines.append("- 目标采样率：`48 kHz`")
    lines.append("- 输出声道：`单声道`")
    lines.append("- 输出格式：`.wav`")
    lines.append("- 切分长度：`1～5 s`")
    lines.append("- 处理流程：转单声道、重采样、去直流、静音裁剪、峰值归一化、切分、可视化")
    lines.append("")

    lines.append("## 2. 总体统计")
    lines.append("")
    lines.append(f"- 扫描原始音频文件数：{total_files}")
    lines.append(f"- 成功清洗完整音频文件数：{processed_files}")
    lines.append(f"- 切分片段总数：{total_segments}")
    lines.append(f"- 有效片段数：{valid_segments}")
    lines.append(f"- 完整音频削波样本总数：{file_clipped_total}")
    lines.append(f"- 片段削波样本总数：{segment_clipped_total}")
    lines.append("")

    if pass_check:
        lines.append("结论：达到今日验收标准，满足 `≥80 段有效片段、无削波、格式统一`。")
    else:
        lines.append("结论：暂未完全达到今日验收标准，需要检查有效片段数量或削波情况。")

    lines.append("")
    lines.append("## 3. 数据集统计")
    lines.append("")
    lines.append(simple_markdown_table(
        ["dataset", "total_files", "processed_files", "valid_segments"],
        dataset_rows,
    ))
    lines.append("")

    lines.append("## 4. 文件处理状态统计")
    lines.append("")
    lines.append(simple_markdown_table(
        ["reason", "count"],
        reason_rows,
    ))
    lines.append("")

    lines.append("## 5. 样例图输出")
    lines.append("")
    if len(example_outputs) == 0:
        lines.append("- 未生成样例图。")
    else:
        for p in example_outputs:
            lines.append(f"- `{p}`")

    lines.append("")
    lines.append("## 6. 输出文件")
    lines.append("")
    lines.append(f"- 文件级质量报告：`{rel_to_root(FILE_REPORT_CSV)}`")
    lines.append(f"- 片段清单：`{rel_to_root(SEGMENT_REPORT_CSV)}`")
    lines.append(f"- Markdown 质量报告：`{rel_to_root(QUALITY_REPORT_MD)}`")
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

    audio_files = find_audio_files(INPUT_ROOT)

    print("========== 第 1 周周三：音频清洗与标准化 ==========")
    print(f"[INFO] 项目根目录: {ROOT}")
    print(f"[INFO] 输入目录: {INPUT_ROOT}")
    print(f"[INFO] 清洗音频输出目录: {PROCESSED_ROOT}")
    print(f"[INFO] 切分片段输出目录: {SEGMENT_ROOT}")
    print(f"[INFO] 质量报告目录: {REPORT_DIR}")
    print(f"[INFO] 找到原始音频文件: {len(audio_files)} 个")
    print("")

    if len(audio_files) == 0:
        print("[WARN] data/raw_sources 下没有找到音频文件，请检查数据是否放对位置。")

    file_records = []
    all_segment_records = []

    for src in tqdm(audio_files, desc="正在清洗与切分音频"):
        file_record, segment_records = process_one_file(src)
        file_records.append(file_record)
        all_segment_records.extend(segment_records)

    file_df = pd.DataFrame(file_records)
    segment_df = pd.DataFrame(all_segment_records)

    file_df.to_csv(FILE_REPORT_CSV, index=False, encoding="utf-8-sig")
    segment_df.to_csv(SEGMENT_REPORT_CSV, index=False, encoding="utf-8-sig")

    example_outputs = generate_example_plots(all_segment_records)

    write_quality_report(file_df, segment_df, example_outputs)

    total_files = len(file_df)
    processed_files = int(file_df["is_processed"].sum()) if total_files > 0 else 0

    total_segments = len(segment_df)
    valid_segments = int(segment_df["is_valid"].sum()) if total_segments > 0 else 0

    file_clipped_total = (
        int(file_df["file_clipped_samples"].fillna(0).sum())
        if total_files > 0 else 0
    )

    segment_clipped_total = (
        int(segment_df["clipped_samples"].fillna(0).sum())
        if total_segments > 0 else 0
    )

    print("")
    print("========== 处理完成 ==========")
    print(f"扫描原始音频文件数: {total_files}")
    print(f"成功清洗完整音频文件数: {processed_files}")
    print(f"切分片段总数: {total_segments}")
    print(f"有效片段数: {valid_segments}")
    print(f"完整音频削波样本总数: {file_clipped_total}")
    print(f"片段削波样本总数: {segment_clipped_total}")
    print("")
    print(f"清洗音频目录: {PROCESSED_ROOT}")
    print(f"切分片段目录: {SEGMENT_ROOT}")
    print(f"文件级质量报告: {FILE_REPORT_CSV}")
    print(f"片段清单: {SEGMENT_REPORT_CSV}")
    print(f"Markdown 质量报告: {QUALITY_REPORT_MD}")
    print(f"样例图目录: {EXAMPLE_DIR}")

    if valid_segments >= 80 and file_clipped_total == 0 and segment_clipped_total == 0:
        print("[PASS] 达到今日验收标准：≥80 段有效片段、无削波、格式统一。")
    else:
        print("[WARN] 暂未完全达到验收标准，请查看质量报告和 CSV 中的 reason 字段。")


if __name__ == "__main__":
    main()