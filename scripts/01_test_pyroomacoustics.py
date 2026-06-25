"""
00_test_pyroomacoustics.py

作用：
1. 测试 pyroomacoustics 是否能正常运行
2. 创建一个简单 3D 房间
3. 放置 1 个测试声源
4. 放置 8 个麦克风
5. 生成多通道麦克风接收信号
6. 保存 test_microphones.wav、test_rir.npy、test_metadata.json

运行方式：
在项目根目录运行：
python scripts/00_test_pyroomacoustics.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pyroomacoustics as pra


def create_test_signal(fs: int, duration: float) -> np.ndarray:
    """
    创建测试声源信号。
    当前用两个正弦波叠加代替鸡叫声。
    后续正式实验会替换成真实鸡叫 wav。
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    signal = (
        0.35 * np.sin(2 * np.pi * 700 * t)
        + 0.15 * np.sin(2 * np.pi * 1200 * t)
    )

    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal = signal / max_abs * 0.6

    return signal.astype(np.float32)


def main():
    # =========================
    # 1. 基础参数
    # =========================
    fs = 16000
    duration = 2.0

    out_dir = Path("results/week1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 2. 创建房间
    # =========================
    room_size = [3.0, 2.0, 2.5]
    energy_absorption = 0.35
    max_order = 3

    # pyroomacoustics 0.10.1 不支持 absorption=...
    # 需要用 materials=Material(...)
    material = pra.Material(energy_absorption=energy_absorption)

    room = pra.ShoeBox(
        p=room_size,
        fs=fs,
        materials=material,
        max_order=max_order,
    )

    # =========================
    # 3. 添加单个声源
    # =========================
    source_signal = create_test_signal(fs=fs, duration=duration)

    source_position = [1.2, 0.9, 1.2]
    room.add_source(source_position, signal=source_signal)

    # =========================
    # 4. 添加 8 个麦克风
    # =========================
    # 每个坐标是 [x, y, z]
    # pyroomacoustics 要求形状为 3 × M
    mic_positions = np.array(
        [
            [0.4, 0.4, 1.2],
            [0.8, 0.4, 1.2],
            [1.2, 0.4, 1.2],
            [1.6, 0.4, 1.2],
            [0.4, 1.6, 1.2],
            [0.8, 1.6, 1.2],
            [1.2, 1.6, 1.2],
            [1.6, 1.6, 1.2],
        ],
        dtype=np.float32,
    ).T

    mic_array = pra.MicrophoneArray(mic_positions, fs=fs)
    room.add_microphone_array(mic_array)

    # =========================
    # 5. 计算 RIR 并仿真
    # =========================
    room.compute_rir()
    room.simulate()

    signals = room.mic_array.signals

    if signals is None or signals.size == 0:
        raise RuntimeError("仿真失败：没有生成麦克风信号。")

    # =========================
    # 6. 保存 8 通道 wav
    # =========================
    # signals 是 M × T
    # soundfile 保存多通道 wav 需要 T × M
    wav_path = out_dir / "test_microphones.wav"
    sf.write(wav_path, signals.T, fs)

    # =========================
    # 7. 保存 RIR
    # =========================
    rir_path = out_dir / "test_rir.npy"
    np.save(rir_path, np.array(room.rir, dtype=object), allow_pickle=True)

    # =========================
    # 8. 保存 metadata
    # =========================
    metadata = {
        "task": "pyroomacoustics_single_source_8_mics_test",
        "description": "单源多麦克风仿真测试，用于验证环境和 Pyroomacoustics 是否正常。",
        "sample_rate": fs,
        "duration_seconds": duration,
        "room_size": room_size,
        "energy_absorption": energy_absorption,
        "max_order": max_order,
        "source_count": 1,
        "source_position": source_position,
        "mic_count": int(mic_positions.shape[1]),
        "mic_positions": mic_positions.T.tolist(),
        "microphone_signal_shape": list(signals.shape),
        "output_wav": str(wav_path),
        "rir_path": str(rir_path),
    }

    metadata_path = out_dir / "test_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # =========================
    # 9. 输出结果
    # =========================
    print("Pyroomacoustics test finished.")
    print(f"Saved wav: {wav_path}")
    print(f"Saved RIR: {rir_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Microphone signals shape: {signals.shape}")
    print("Test passed.")


if __name__ == "__main__":
    main()