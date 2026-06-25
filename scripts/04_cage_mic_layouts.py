# -*- coding: utf-8 -*-
"""
第 1 周第 4 天：坐标系与麦克风布局

论文级基准场景版本。

功能：
1. 定义鸡笼局部三维坐标系；
2. 生成 cage.yaml；
3. 设计 4 / 6 / 8 / 12 麦克风数量布局；
4. 设计 E2 所需的四角 / 两侧 / 单边 / 集中阵列布局；
5. 生成 mic_layouts.yaml；
6. 检查所有麦克风坐标是否合法；
7. 绘制俯视布局图；
8. 输出布局合法性报告。

坐标系约定：
- 单位：米；
- 原点 O = 鸡笼左前下角；
- x 轴：沿鸡笼长度方向，向右为正；
- y 轴：沿鸡笼宽度方向，向后为正；
- z 轴：沿鸡笼高度方向，向上为正。

正式基准参数：
- 鸡笼局部尺寸：1.20 × 0.75 × 0.60 m；
- 发声平面：z = 0.35 m；
- 发声高度实验值：0.25 / 0.35 / 0.45 m；
- 基准麦克风数：8；
- 基准布局：鸡笼四周分布式；
- 采样率：48 kHz；
- 基准 RT60：0.30 s；
- 基准 SNR：20 dB；
- 网格间距：2 cm；
- STFT：2048 点，hop=512。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 路径配置
# =========================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "configs" / "week1" / "day4"
RESULT_DIR = ROOT / "results" / "week1" / "day4" / "layouts"

CAGE_YAML = CONFIG_DIR / "cage.yaml"
MIC_LAYOUTS_YAML = CONFIG_DIR / "mic_layouts.yaml"

MIC_COUNT_LAYOUT_FIG = RESULT_DIR / "mic_count_layouts_top_view.png"
LAYOUT_VARIANTS_FIG = RESULT_DIR / "layout_variants_top_view.png"
VALIDATION_REPORT = RESULT_DIR / "layout_validation_report.md"


# =========================
# 正式基准参数
# =========================

CAGE_LENGTH_X = 1.20
CAGE_WIDTH_Y = 0.75
CAGE_HEIGHT_Z = 0.60

# 发声平面：后续仿真脚本 05 / 06 要读取这个值
SOURCE_PLANE_Z = 0.35
SOURCE_HEIGHT_VALUES = [0.25, 0.35, 0.45]

# 麦克风安装高度：必须小于鸡笼高度 0.60 m
MIC_HEIGHT_Z = 0.50

# 麦克风离边界留一点距离，避免贴墙
MARGIN_X = 0.08
MARGIN_Y = 0.08

# 常用坐标
X_LEFT = MARGIN_X
X_CENTER = CAGE_LENGTH_X / 2.0
X_RIGHT = CAGE_LENGTH_X - MARGIN_X

Y_FRONT = MARGIN_Y
Y_CENTER = CAGE_WIDTH_Y / 2.0
Y_BACK = CAGE_WIDTH_Y - MARGIN_Y


# =========================
# 鸡笼局部坐标配置
# =========================

def build_cage_config() -> Dict:
    """构建鸡笼局部坐标系配置。"""
    cage = {
        "name": "local_chicken_cage_paper_baseline",
        "unit": "meter",
        "coordinate_system": {
            "origin": "left_front_bottom_corner",
            "x_axis": "length_direction_left_to_right",
            "y_axis": "width_direction_front_to_back",
            "z_axis": "height_direction_bottom_to_top",
            "description": "局部坐标系原点位于鸡笼左前下角，单位为米。"
        },
        "dimensions": {
            "length_x": CAGE_LENGTH_X,
            "width_y": CAGE_WIDTH_Y,
            "height_z": CAGE_HEIGHT_Z
        },
        "valid_range": {
            "x": [0.0, CAGE_LENGTH_X],
            "y": [0.0, CAGE_WIDTH_Y],
            "z": [0.0, CAGE_HEIGHT_Z]
        },
        "source_plane": {
            "baseline_z": SOURCE_PLANE_Z,
            "height_error_values": SOURCE_HEIGHT_VALUES,
            "description": "后续仿真实验默认发声平面为 z=0.35 m，高度误差实验使用 0.25/0.35/0.45 m。"
        },
        "baseline_experiment_params": {
            "fs": 48000,
            "baseline_num_mics": 8,
            "baseline_layout": "mic_8",
            "baseline_layout_description": "鸡笼四周分布式 8 麦克风布局",
            "baseline_rt60_sec": 0.30,
            "rt60_values_sec": [0.10, 0.30, 0.50, 0.70],
            "baseline_snr_db": 20,
            "snr_values_db": [0, 5, 10, 20, 30],
            "baseline_grid_spacing_m": 0.02,
            "grid_spacing_values_m": [0.01, 0.02, 0.05],
            "stft_n_fft": 2048,
            "stft_hop_length": 512,
            "stft_compare": {
                "n_fft": 1024,
                "hop_length": 256
            }
        },
        "reference_points": {
            "left_front_bottom": [0.0, 0.0, 0.0],
            "right_front_bottom": [CAGE_LENGTH_X, 0.0, 0.0],
            "left_back_bottom": [0.0, CAGE_WIDTH_Y, 0.0],
            "right_back_bottom": [CAGE_LENGTH_X, CAGE_WIDTH_Y, 0.0],
            "left_front_top": [0.0, 0.0, CAGE_HEIGHT_Z],
            "right_front_top": [CAGE_LENGTH_X, 0.0, CAGE_HEIGHT_Z],
            "left_back_top": [0.0, CAGE_WIDTH_Y, CAGE_HEIGHT_Z],
            "right_back_top": [CAGE_LENGTH_X, CAGE_WIDTH_Y, CAGE_HEIGHT_Z]
        },
        "notes": [
            "本配置用于第1周第4天坐标系与麦克风布局实验。",
            "鸡笼局部尺寸固定为 1.20 × 0.75 × 0.60 m。",
            "所有麦克风坐标必须位于 valid_range 指定范围内。",
            "后续 Pyroomacoustics 仿真必须读取本文件中的 dimensions、source_plane 和 baseline_experiment_params。"
        ]
    }
    return cage


# =========================
# 布局工具函数
# =========================

def mic(mic_id: str, x: float, y: float, z: float = MIC_HEIGHT_Z) -> Dict:
    """生成单个麦克风字典。"""
    return {
        "id": mic_id,
        "position": [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
    }


def make_layout(
    name: str,
    description: str,
    microphones: List[Dict],
    layout_family: str,
    variable_for: str
) -> Dict:
    """生成布局配置。"""
    return {
        "name": name,
        "description": description,
        "layout_family": layout_family,
        "variable_for": variable_for,
        "height_z": MIC_HEIGHT_Z,
        "expected_mic_count": len(microphones),
        "microphones": microphones
    }


# =========================
# 麦克风布局设计
# =========================

def build_mic_layouts() -> Dict:
    """
    构建论文级实验需要的麦克风布局。

    包含两类：
    1. E1 麦克风数量实验：mic_4 / mic_6 / mic_8 / mic_12；
    2. E2 布局实验：四角 / 两侧 / 单边 / 集中阵列，均以 8 麦为基准。
    """

    # E1：不同麦克风数量
    mic_4 = [
        mic("M01", X_LEFT, Y_FRONT),
        mic("M02", X_RIGHT, Y_FRONT),
        mic("M03", X_LEFT, Y_BACK),
        mic("M04", X_RIGHT, Y_BACK),
    ]

    mic_6 = [
        mic("M01", X_LEFT, Y_FRONT),
        mic("M02", X_CENTER, Y_FRONT),
        mic("M03", X_RIGHT, Y_FRONT),
        mic("M04", X_LEFT, Y_BACK),
        mic("M05", X_CENTER, Y_BACK),
        mic("M06", X_RIGHT, Y_BACK),
    ]

    # 基准 8 麦：鸡笼四周分布式
    mic_8 = [
        mic("M01", X_LEFT, Y_FRONT),
        mic("M02", X_CENTER, Y_FRONT),
        mic("M03", X_RIGHT, Y_FRONT),
        mic("M04", X_LEFT, Y_CENTER),
        mic("M05", X_RIGHT, Y_CENTER),
        mic("M06", X_LEFT, Y_BACK),
        mic("M07", X_CENTER, Y_BACK),
        mic("M08", X_RIGHT, Y_BACK),
    ]

    mic_12 = [
        mic("M01", X_LEFT, Y_FRONT),
        mic("M02", 0.43, Y_FRONT),
        mic("M03", 0.77, Y_FRONT),
        mic("M04", X_RIGHT, Y_FRONT),

        mic("M05", X_LEFT, Y_CENTER),
        mic("M06", 0.43, Y_CENTER),
        mic("M07", 0.77, Y_CENTER),
        mic("M08", X_RIGHT, Y_CENTER),

        mic("M09", X_LEFT, Y_BACK),
        mic("M10", 0.43, Y_BACK),
        mic("M11", 0.77, Y_BACK),
        mic("M12", X_RIGHT, Y_BACK),
    ]

    # E2：布局实验，固定 8 麦
    layout_corners_8 = [
        mic("M01", X_LEFT, Y_FRONT),
        mic("M02", 0.25, Y_FRONT),
        mic("M03", 0.95, Y_FRONT),
        mic("M04", X_RIGHT, Y_FRONT),
        mic("M05", X_LEFT, Y_BACK),
        mic("M06", 0.25, Y_BACK),
        mic("M07", 0.95, Y_BACK),
        mic("M08", X_RIGHT, Y_BACK),
    ]

    layout_sides_8 = [
        mic("M01", X_LEFT, Y_FRONT),
        mic("M02", 0.43, Y_FRONT),
        mic("M03", 0.77, Y_FRONT),
        mic("M04", X_RIGHT, Y_FRONT),
        mic("M05", X_LEFT, Y_BACK),
        mic("M06", 0.43, Y_BACK),
        mic("M07", 0.77, Y_BACK),
        mic("M08", X_RIGHT, Y_BACK),
    ]

    layout_single_side_8 = [
        mic("M01", 0.08, Y_FRONT),
        mic("M02", 0.23, Y_FRONT),
        mic("M03", 0.38, Y_FRONT),
        mic("M04", 0.53, Y_FRONT),
        mic("M05", 0.68, Y_FRONT),
        mic("M06", 0.83, Y_FRONT),
        mic("M07", 0.98, Y_FRONT),
        mic("M08", 1.12, Y_FRONT),
    ]

    layout_compact_8 = [
        mic("M01", 0.48, 0.30),
        mic("M02", 0.56, 0.30),
        mic("M03", 0.64, 0.30),
        mic("M04", 0.72, 0.30),
        mic("M05", 0.48, 0.45),
        mic("M06", 0.56, 0.45),
        mic("M07", 0.64, 0.45),
        mic("M08", 0.72, 0.45),
    ]

    layouts = {
        "unit": "meter",
        "coordinate_order": ["x", "y", "z"],
        "baseline_layout": "mic_8",
        "baseline_description": "鸡笼四周分布式 8 麦克风布局",
        "notes": [
            "mic_4/mic_6/mic_8/mic_12 用于 E1 麦克风数量实验。",
            "layout_corners_8/layout_sides_8/layout_single_side_8/layout_compact_8 用于 E2 布局实验。",
            "所有布局均位于 1.20 × 0.75 × 0.60 m 鸡笼局部坐标范围内。",
            "所有麦克风默认安装高度为 z=0.50 m。"
        ],
        "layouts": {
            "mic_4": make_layout(
                "mic_4",
                "4 麦克风四角布局，用于 E1 麦克风数量实验。",
                mic_4,
                layout_family="mic_count",
                variable_for="E1_num_mics"
            ),
            "mic_6": make_layout(
                "mic_6",
                "6 麦克风两侧三点布局，用于 E1 麦克风数量实验。",
                mic_6,
                layout_family="mic_count",
                variable_for="E1_num_mics"
            ),
            "mic_8": make_layout(
                "mic_8",
                "基准 8 麦克风鸡笼四周分布式布局。",
                mic_8,
                layout_family="mic_count",
                variable_for="E1_num_mics_and_baseline"
            ),
            "mic_12": make_layout(
                "mic_12",
                "12 麦克风三排四列分布式布局，用于 E1 麦克风数量实验。",
                mic_12,
                layout_family="mic_count",
                variable_for="E1_num_mics"
            ),
            "layout_corners_8": make_layout(
                "layout_corners_8",
                "8 麦克风四角强化布局，用于 E2 布局实验。",
                layout_corners_8,
                layout_family="layout_variant",
                variable_for="E2_layout"
            ),
            "layout_sides_8": make_layout(
                "layout_sides_8",
                "8 麦克风两侧分布式布局，用于 E2 布局实验。",
                layout_sides_8,
                layout_family="layout_variant",
                variable_for="E2_layout"
            ),
            "layout_single_side_8": make_layout(
                "layout_single_side_8",
                "8 麦克风单边线阵布局，用于 E2 布局实验。",
                layout_single_side_8,
                layout_family="layout_variant",
                variable_for="E2_layout"
            ),
            "layout_compact_8": make_layout(
                "layout_compact_8",
                "8 麦克风中心集中阵列布局，用于 E2 布局实验。",
                layout_compact_8,
                layout_family="layout_variant",
                variable_for="E2_layout"
            ),
        }
    }

    return layouts


# =========================
# YAML 读写函数
# =========================

def save_yaml(data: Dict, path: Path):
    """保存 YAML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )


def load_yaml(path: Path) -> Dict:
    """读取 YAML 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# 坐标合法性检查
# =========================

def is_position_valid(position: List[float], valid_range: Dict) -> Tuple[bool, str]:
    """检查单个三维坐标是否合法。"""
    if not isinstance(position, list) or len(position) != 3:
        return False, "position 必须是长度为 3 的列表 [x, y, z]"

    x, y, z = position

    x_min, x_max = valid_range["x"]
    y_min, y_max = valid_range["y"]
    z_min, z_max = valid_range["z"]

    if not (x_min <= x <= x_max):
        return False, f"x={x} 超出范围 [{x_min}, {x_max}]"
    if not (y_min <= y <= y_max):
        return False, f"y={y} 超出范围 [{y_min}, {y_max}]"
    if not (z_min <= z <= z_max):
        return False, f"z={z} 超出范围 [{z_min}, {z_max}]"

    return True, "ok"


def validate_layouts(cage: Dict, mic_layouts: Dict) -> List[Dict]:
    """检查所有布局中所有麦克风坐标是否合法。"""
    valid_range = cage["valid_range"]
    results = []

    for layout_name, layout_info in mic_layouts["layouts"].items():
        microphones = layout_info["microphones"]
        expected_count = int(layout_info["expected_mic_count"])
        actual_count = len(microphones)

        if actual_count != expected_count:
            results.append({
                "layout": layout_name,
                "mic_id": "-",
                "position": "-",
                "is_valid": False,
                "message": f"麦克风数量错误：期望 {expected_count}，实际 {actual_count}"
            })

        seen_ids = set()
        seen_positions = set()

        for mic_info in microphones:
            mic_id = mic_info["id"]
            position = mic_info["position"]
            pos_key = tuple(position)

            if mic_id in seen_ids:
                results.append({
                    "layout": layout_name,
                    "mic_id": mic_id,
                    "position": position,
                    "is_valid": False,
                    "message": "麦克风 ID 重复"
                })
                continue

            if pos_key in seen_positions:
                results.append({
                    "layout": layout_name,
                    "mic_id": mic_id,
                    "position": position,
                    "is_valid": False,
                    "message": "麦克风坐标重复"
                })
                continue

            seen_ids.add(mic_id)
            seen_positions.add(pos_key)

            ok, message = is_position_valid(position, valid_range)

            results.append({
                "layout": layout_name,
                "mic_id": mic_id,
                "position": position,
                "is_valid": ok,
                "message": message
            })

    return results


# =========================
# 绘图函数
# =========================

def draw_cage_boundary(ax, cage: Dict):
    """绘制鸡笼俯视边界。"""
    length_x = cage["dimensions"]["length_x"]
    width_y = cage["dimensions"]["width_y"]

    xs = [0, length_x, length_x, 0, 0]
    ys = [0, 0, width_y, width_y, 0]

    ax.plot(xs, ys, linewidth=2)
    ax.set_xlim(-0.08, length_x + 0.08)
    ax.set_ylim(-0.08, width_y + 0.08)
    ax.set_aspect("equal", adjustable="box")


def plot_layout_group(
    cage: Dict,
    mic_layouts: Dict,
    layout_names: List[str],
    out_path: Path,
    title: str
):
    """绘制指定布局组的俯视图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_layouts = len(layout_names)

    if num_layouts <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 4

    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, layout_name in zip(axes, layout_names):
        ax.axis("on")

        layout = mic_layouts["layouts"][layout_name]
        microphones = layout["microphones"]

        draw_cage_boundary(ax, cage)

        xs = [m["position"][0] for m in microphones]
        ys = [m["position"][1] for m in microphones]

        ax.scatter(xs, ys, s=70)

        for m in microphones:
            mic_id = m["id"]
            x, y, z = m["position"]
            ax.text(x + 0.015, y + 0.015, mic_id, fontsize=8)

        ax.set_title(f"{layout_name}: {len(microphones)} microphones")
        ax.set_xlabel("x / length direction (m)")
        ax.set_ylabel("y / width direction (m)")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# 报告函数
# =========================

def write_validation_report(cage: Dict, mic_layouts: Dict, validation_results: List[Dict]):
    """写出 Markdown 合法性检查报告。"""
    total = len(validation_results)
    valid_count = sum(1 for r in validation_results if r["is_valid"])
    invalid_count = total - valid_count

    lines = []

    lines.append("# 第 1 周第 4 天：坐标系与麦克风布局检查报告")
    lines.append("")
    lines.append("## 1. 鸡笼局部坐标系")
    lines.append("")
    lines.append(f"- 单位：`{cage['unit']}`")
    lines.append("- 原点：鸡笼左前下角")
    lines.append("- x 轴：鸡笼长度方向，向右为正")
    lines.append("- y 轴：鸡笼宽度方向，向后为正")
    lines.append("- z 轴：鸡笼高度方向，向上为正")
    lines.append("")
    lines.append("鸡笼尺寸：")
    lines.append("")
    lines.append(f"- length_x：`{cage['dimensions']['length_x']} m`")
    lines.append(f"- width_y：`{cage['dimensions']['width_y']} m`")
    lines.append(f"- height_z：`{cage['dimensions']['height_z']} m`")
    lines.append("")
    lines.append("发声平面：")
    lines.append("")
    lines.append(f"- baseline_z：`{cage['source_plane']['baseline_z']} m`")
    lines.append(f"- height_error_values：`{cage['source_plane']['height_error_values']}`")
    lines.append("")

    lines.append("## 2. 基准实验参数")
    lines.append("")
    params = cage["baseline_experiment_params"]
    lines.append(f"- 采样率：`{params['fs']} Hz`")
    lines.append(f"- 基准麦克风数：`{params['baseline_num_mics']}`")
    lines.append(f"- 基准布局：`{params['baseline_layout']}`")
    lines.append(f"- 基准 RT60：`{params['baseline_rt60_sec']} s`")
    lines.append(f"- RT60 实验取值：`{params['rt60_values_sec']}`")
    lines.append(f"- 基准 SNR：`{params['baseline_snr_db']} dB`")
    lines.append(f"- SNR 实验取值：`{params['snr_values_db']}`")
    lines.append(f"- 基准网格间距：`{params['baseline_grid_spacing_m']} m`")
    lines.append(f"- 网格间距实验取值：`{params['grid_spacing_values_m']}`")
    lines.append(f"- STFT：`n_fft={params['stft_n_fft']}, hop={params['stft_hop_length']}`")
    lines.append("")

    lines.append("## 3. 麦克风布局")
    lines.append("")
    for layout_name, layout_info in mic_layouts["layouts"].items():
        lines.append(f"### {layout_name}")
        lines.append("")
        lines.append(f"- 描述：{layout_info['description']}")
        lines.append(f"- 布局类型：`{layout_info['layout_family']}`")
        lines.append(f"- 对应变量：`{layout_info['variable_for']}`")
        lines.append(f"- 麦克风数量：{len(layout_info['microphones'])}")
        lines.append(f"- 默认高度：`{layout_info['height_z']} m`")
        lines.append("")
        lines.append("| mic_id | x | y | z |")
        lines.append("|---|---:|---:|---:|")
        for mic_info in layout_info["microphones"]:
            x, y, z = mic_info["position"]
            lines.append(f"| {mic_info['id']} | {x:.3f} | {y:.3f} | {z:.3f} |")
        lines.append("")

    lines.append("## 4. 坐标合法性检查")
    lines.append("")
    lines.append(f"- 检查记录数：{total}")
    lines.append(f"- 合法记录数：{valid_count}")
    lines.append(f"- 非法记录数：{invalid_count}")
    lines.append("")

    if invalid_count == 0:
        lines.append("结论：所有麦克风坐标均位于鸡笼合法范围内，YAML 配置可被脚本正常读取。")
    else:
        lines.append("结论：存在非法坐标，需要检查下表。")

    lines.append("")
    lines.append("| layout | mic_id | position | is_valid | message |")
    lines.append("|---|---|---|---|---|")

    for r in validation_results:
        lines.append(
            f"| {r['layout']} | {r['mic_id']} | {r['position']} | "
            f"{r['is_valid']} | {r['message']} |"
        )

    lines.append("")
    lines.append("## 5. 输出文件")
    lines.append("")
    lines.append(f"- 鸡笼配置：`{CAGE_YAML.relative_to(ROOT)}`")
    lines.append(f"- 麦克风布局配置：`{MIC_LAYOUTS_YAML.relative_to(ROOT)}`")
    lines.append(f"- 麦克风数量布局图：`{MIC_COUNT_LAYOUT_FIG.relative_to(ROOT)}`")
    lines.append(f"- 布局变量图：`{LAYOUT_VARIANTS_FIG.relative_to(ROOT)}`")
    lines.append(f"- 检查报告：`{VALIDATION_REPORT.relative_to(ROOT)}`")
    lines.append("")

    VALIDATION_REPORT.write_text("\n".join(lines), encoding="utf-8")


# =========================
# 主函数
# =========================

def main():
    """主函数：生成配置、读取配置、检查合法性、绘制布局图。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("========== 第 1 周第 4 天：坐标系与麦克风布局 ==========")
    print(f"[INFO] 项目根目录: {ROOT}")
    print(f"[INFO] 配置目录: {CONFIG_DIR}")
    print(f"[INFO] 结果目录: {RESULT_DIR}")

    # 1. 生成配置文件
    cage_config = build_cage_config()
    mic_layouts_config = build_mic_layouts()

    save_yaml(cage_config, CAGE_YAML)
    save_yaml(mic_layouts_config, MIC_LAYOUTS_YAML)

    print(f"[OK] 已生成: {CAGE_YAML}")
    print(f"[OK] 已生成: {MIC_LAYOUTS_YAML}")

    # 2. 重新读取 YAML，验证配置可被脚本读取
    cage = load_yaml(CAGE_YAML)
    mic_layouts = load_yaml(MIC_LAYOUTS_YAML)

    print("[OK] YAML 配置读取成功")

    # 3. 合法性检查
    validation_results = validate_layouts(cage, mic_layouts)
    invalid_results = [r for r in validation_results if not r["is_valid"]]

    if len(invalid_results) == 0:
        print("[PASS] 所有麦克风坐标均合法")
    else:
        print("[WARN] 存在非法麦克风坐标")
        for r in invalid_results:
            print(r)

    # 4. 绘制 E1 麦克风数量布局图
    plot_layout_group(
        cage,
        mic_layouts,
        ["mic_4", "mic_6", "mic_8", "mic_12"],
        MIC_COUNT_LAYOUT_FIG,
        "E1 Microphone Count Layouts: 4 / 6 / 8 / 12"
    )
    print(f"[OK] 已生成麦克风数量布局图: {MIC_COUNT_LAYOUT_FIG}")

    # 5. 绘制 E2 布局变量图
    plot_layout_group(
        cage,
        mic_layouts,
        [
            "layout_corners_8",
            "layout_sides_8",
            "layout_single_side_8",
            "layout_compact_8",
        ],
        LAYOUT_VARIANTS_FIG,
        "E2 Layout Variants: Corners / Sides / Single Side / Compact"
    )
    print(f"[OK] 已生成布局变量图: {LAYOUT_VARIANTS_FIG}")

    # 6. 写报告
    write_validation_report(cage, mic_layouts, validation_results)
    print(f"[OK] 已生成检查报告: {VALIDATION_REPORT}")

    print("")
    print("========== 处理完成 ==========")
    print(f"鸡笼配置: {CAGE_YAML}")
    print(f"麦克风布局配置: {MIC_LAYOUTS_YAML}")
    print(f"麦克风数量布局图: {MIC_COUNT_LAYOUT_FIG}")
    print(f"布局变量图: {LAYOUT_VARIANTS_FIG}")
    print(f"检查报告: {VALIDATION_REPORT}")

    if len(invalid_results) == 0:
        print("[PASS] 今日验收标准通过：坐标均合法，配置可被脚本读取。")
    else:
        print("[WARN] 今日验收标准未通过：存在非法坐标。")


if __name__ == "__main__":
    main()