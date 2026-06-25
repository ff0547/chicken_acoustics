# -*- coding: utf-8 -*-
"""
第 1 周第 4 天：坐标系与麦克风布局

功能：
1. 定义鸡笼局部三维坐标系
2. 生成 cage.yaml
3. 设计 4 / 6 / 8 / 12 麦克风布局
4. 生成 mic_layouts.yaml
5. 检查所有麦克风坐标是否合法
6. 绘制俯视布局图
7. 输出布局合法性报告

坐标系约定：
- 单位：米
- 原点 O = 鸡笼左前下角
- x 轴：沿鸡笼长度方向，向右为正
- y 轴：沿鸡笼宽度方向，向后为正
- z 轴：沿鸡笼高度方向，向上为正
"""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np

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

LAYOUT_FIG = RESULT_DIR / "mic_layouts_top_view.png"
VALIDATION_REPORT = RESULT_DIR / "layout_validation_report.md"


# =========================
# 鸡笼局部坐标配置
# =========================

def build_cage_config() -> Dict:
    """
    构建鸡笼局部坐标系配置。

    这里使用一个局部小型鸡笼/鸡舍区域作为实验空间。
    后续如果实际尺寸变化，只需要改 length / width / height。
    """
    cage = {
        "name": "local_chicken_cage",
        "unit": "meter",
        "coordinate_system": {
            "origin": "left_front_bottom_corner",
            "x_axis": "length_direction_left_to_right",
            "y_axis": "width_direction_front_to_back",
            "z_axis": "height_direction_bottom_to_top",
            "description": "局部坐标系原点位于鸡笼左前下角，单位为米。"
        },
        "dimensions": {
            "length_x": 2.40,
            "width_y": 1.20,
            "height_z": 1.80
        },
        "valid_range": {
            "x": [0.0, 2.40],
            "y": [0.0, 1.20],
            "z": [0.0, 1.80]
        },
        "reference_points": {
            "left_front_bottom": [0.0, 0.0, 0.0],
            "right_front_bottom": [2.40, 0.0, 0.0],
            "left_back_bottom": [0.0, 1.20, 0.0],
            "right_back_bottom": [2.40, 1.20, 0.0],
            "left_front_top": [0.0, 0.0, 1.80],
            "right_front_top": [2.40, 0.0, 1.80],
            "left_back_top": [0.0, 1.20, 1.80],
            "right_back_top": [2.40, 1.20, 1.80]
        },
        "notes": [
            "本配置用于第1周第4天坐标系与麦克风布局实验。",
            "所有麦克风坐标必须位于 valid_range 指定范围内。",
            "后续 Pyroomacoustics 仿真可直接读取 dimensions 与 mic_layouts。"
        ]
    }
    return cage


# =========================
# 麦克风布局设计
# =========================

def build_mic_layouts() -> Dict:
    """
    构建 4 / 6 / 8 / 12 麦克风布局。

    设计原则：
    - 所有麦克风均位于鸡笼内部合法范围；
    - 默认高度 z = 1.20 m，略高于鸡只主要发声区域；
    - 4 麦：四角布局，适合基础定位；
    - 6 麦：前后两排各 3 个，提高长度方向分辨率；
    - 8 麦：四角 + 边中点，提高空间覆盖；
    - 12 麦：三排四列，提高二维定位与后续波束形成稳定性。
    """
    layouts = {
        "unit": "meter",
        "coordinate_order": ["x", "y", "z"],
        "layouts": {
            "mic_4": {
                "description": "4麦克风四角布局，用于基础声源定位实验。",
                "height_z": 1.20,
                "microphones": [
                    {"id": "M01", "position": [0.20, 0.20, 1.20]},
                    {"id": "M02", "position": [2.20, 0.20, 1.20]},
                    {"id": "M03", "position": [0.20, 1.00, 1.20]},
                    {"id": "M04", "position": [2.20, 1.00, 1.20]},
                ]
            },
            "mic_6": {
                "description": "6麦克风两排三列布局，增强长度方向定位能力。",
                "height_z": 1.20,
                "microphones": [
                    {"id": "M01", "position": [0.20, 0.20, 1.20]},
                    {"id": "M02", "position": [1.20, 0.20, 1.20]},
                    {"id": "M03", "position": [2.20, 0.20, 1.20]},
                    {"id": "M04", "position": [0.20, 1.00, 1.20]},
                    {"id": "M05", "position": [1.20, 1.00, 1.20]},
                    {"id": "M06", "position": [2.20, 1.00, 1.20]},
                ]
            },
            "mic_8": {
                "description": "8麦克风四角加边中点布局，兼顾覆盖范围和定位稳定性。",
                "height_z": 1.20,
                "microphones": [
                    {"id": "M01", "position": [0.20, 0.20, 1.20]},
                    {"id": "M02", "position": [1.20, 0.20, 1.20]},
                    {"id": "M03", "position": [2.20, 0.20, 1.20]},
                    {"id": "M04", "position": [0.20, 0.60, 1.20]},
                    {"id": "M05", "position": [2.20, 0.60, 1.20]},
                    {"id": "M06", "position": [0.20, 1.00, 1.20]},
                    {"id": "M07", "position": [1.20, 1.00, 1.20]},
                    {"id": "M08", "position": [2.20, 1.00, 1.20]},
                ]
            },
            "mic_12": {
                "description": "12麦克风三排四列布局，用于更密集的二维定位和后续波束形成实验。",
                "height_z": 1.20,
                "microphones": [
                    {"id": "M01", "position": [0.20, 0.20, 1.20]},
                    {"id": "M02", "position": [0.80, 0.20, 1.20]},
                    {"id": "M03", "position": [1.60, 0.20, 1.20]},
                    {"id": "M04", "position": [2.20, 0.20, 1.20]},

                    {"id": "M05", "position": [0.20, 0.60, 1.20]},
                    {"id": "M06", "position": [0.80, 0.60, 1.20]},
                    {"id": "M07", "position": [1.60, 0.60, 1.20]},
                    {"id": "M08", "position": [2.20, 0.60, 1.20]},

                    {"id": "M09", "position": [0.20, 1.00, 1.20]},
                    {"id": "M10", "position": [0.80, 1.00, 1.20]},
                    {"id": "M11", "position": [1.60, 1.00, 1.20]},
                    {"id": "M12", "position": [2.20, 1.00, 1.20]},
                ]
            }
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
        expected_count = int(layout_name.split("_")[1])
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

        for mic in microphones:
            mic_id = mic["id"]
            position = mic["position"]

            if mic_id in seen_ids:
                results.append({
                    "layout": layout_name,
                    "mic_id": mic_id,
                    "position": position,
                    "is_valid": False,
                    "message": "麦克风 ID 重复"
                })
                continue

            seen_ids.add(mic_id)

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
    ax.set_xlim(-0.15, length_x + 0.15)
    ax.set_ylim(-0.15, width_y + 0.15)
    ax.set_aspect("equal", adjustable="box")


def plot_layouts_top_view(cage: Dict, mic_layouts: Dict, out_path: Path):
    """绘制 4 / 6 / 8 / 12 麦克风俯视布局图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layouts = mic_layouts["layouts"]
    layout_names = ["mic_4", "mic_6", "mic_8", "mic_12"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, layout_name in zip(axes, layout_names):
        layout = layouts[layout_name]
        microphones = layout["microphones"]

        draw_cage_boundary(ax, cage)

        xs = [m["position"][0] for m in microphones]
        ys = [m["position"][1] for m in microphones]

        ax.scatter(xs, ys, s=70)

        for m in microphones:
            mic_id = m["id"]
            x, y, z = m["position"]
            ax.text(x + 0.03, y + 0.03, mic_id, fontsize=9)

        ax.set_title(f"{layout_name}: {len(microphones)} microphones")
        ax.set_xlabel("x / length direction (m)")
        ax.set_ylabel("y / width direction (m)")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Top View of Microphone Layouts in Local Cage Coordinate System")
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

    lines.append("## 2. 麦克风布局")
    lines.append("")
    for layout_name, layout_info in mic_layouts["layouts"].items():
        lines.append(f"### {layout_name}")
        lines.append("")
        lines.append(f"- 描述：{layout_info['description']}")
        lines.append(f"- 麦克风数量：{len(layout_info['microphones'])}")
        lines.append(f"- 默认高度：`{layout_info['height_z']} m`")
        lines.append("")
        lines.append("| mic_id | x | y | z |")
        lines.append("|---|---:|---:|---:|")
        for mic in layout_info["microphones"]:
            x, y, z = mic["position"]
            lines.append(f"| {mic['id']} | {x:.2f} | {y:.2f} | {z:.2f} |")
        lines.append("")

    lines.append("## 3. 坐标合法性检查")
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
    lines.append("## 4. 输出文件")
    lines.append("")
    lines.append(f"- 鸡笼配置：`{CAGE_YAML.relative_to(ROOT)}`")
    lines.append(f"- 麦克风布局配置：`{MIC_LAYOUTS_YAML.relative_to(ROOT)}`")
    lines.append(f"- 俯视布局图：`{LAYOUT_FIG.relative_to(ROOT)}`")
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

    # 4. 绘制俯视布局图
    plot_layouts_top_view(cage, mic_layouts, LAYOUT_FIG)
    print(f"[OK] 已生成俯视布局图: {LAYOUT_FIG}")

    # 5. 写报告
    write_validation_report(cage, mic_layouts, validation_results)
    print(f"[OK] 已生成检查报告: {VALIDATION_REPORT}")

    print("")
    print("========== 处理完成 ==========")
    print(f"鸡笼配置: {CAGE_YAML}")
    print(f"麦克风布局配置: {MIC_LAYOUTS_YAML}")
    print(f"布局图: {LAYOUT_FIG}")
    print(f"检查报告: {VALIDATION_REPORT}")

    if len(invalid_results) == 0:
        print("[PASS] 今日验收标准通过：坐标均合法，配置可被脚本读取。")
    else:
        print("[WARN] 今日验收标准未通过：存在非法坐标。")


if __name__ == "__main__":
    main()