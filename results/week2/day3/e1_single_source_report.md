# 第 2 周 Day3：E1 单源批量定位实验报告

## 1. 实验目的

本实验对应论文级实验矩阵 E1：单源定位中的麦克风数量影响实验。
通过比较 4 / 6 / 8 / 12 个麦克风条件下的二维定位误差，评估麦克风数量对 GCC-PHAT + 近场 SRP-PHAT 定位精度的影响。

## 2. 实验设置

- 鸡笼局部尺寸：`1.2 × 0.75 × 0.6 m`
- 坐标系：`x` 为长度方向，`y` 为深度方向，`z` 为高度方向
- 发声平面：`z = 0.35 m`
- 搜索平面：`z = 0.35 m`
- 麦克风数量：`4 / 6 / 8 / 12`
- 每组场景数：`100`
- 总定位次数：`400`
- 活动源数：`K = 1`
- 场景类型：`单源、静止声源、几何声学混响、加性白噪声`
- RT60 目标值：`0.3 s`
- SNR：`20.0 dB`
- 采样率：`48000 Hz`
- 声速：`343.0 m/s`
- 网格间距：`0.02 m`
- GCC-PHAT 插值倍数：`16`
- TDOA 取分窗口：`±2.0 samples`
- 早期分析窗口：`0.22 s`
- Pyroomacoustics 吸声系数：`0.070049`
- inverse_sabine 返回 max_order：`219`
- 实际使用 max_order：`12`

## 3. 鲁棒化处理

- GCC-PHAT 响应正值化与归一化。
- 仅使用早期分析窗口以降低混响尾部影响。
- 在候选 TDOA 附近小窗口取最大响应，缓解采样量化误差。
- 使用麦克风对可靠性权重降低假峰严重麦对的影响。
- 使用麦克风基线长度权重增强空间分辨力。
- 对 SRP 得分图做 3×3 平滑，压制孤立假峰。

## 4. E1 统计结果

| experiment_id | variable | value | mic_layout | num_mics | n | mean_error_cm | std_error_cm | median_error_cm | p90_error_cm | hit_rate_10cm | hit_rate_20cm | ci95_error_cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | mic_count | 4 | mic_4 | 4 | 100 | 14.2926 | 16.9185 | 8.0674 | 37.7105 | 0.6300 | 0.8000 | 3.3160 |
| E1 | mic_count | 6 | mic_6 | 6 | 100 | 8.3404 | 8.0716 | 6.3566 | 13.0867 | 0.7800 | 0.9400 | 1.5820 |
| E1 | mic_count | 8 | mic_8 | 8 | 100 | 5.2595 | 4.1004 | 4.1490 | 9.6063 | 0.9200 | 0.9900 | 0.8037 |
| E1 | mic_count | 12 | mic_12 | 12 | 100 | 2.6118 | 2.6213 | 1.6429 | 7.9057 | 0.9800 | 1.0000 | 0.5138 |

## 5. Day3 验收结论

- 基准配置：`mic_8`
- 基准配置平均误差：`5.259 cm`
- 验收标准：`mean_error_cm <= 10.0 cm`
- 验收结论：`通过`

## 6. 仿真边界说明

本阶段仿真采用 Pyroomacoustics 的几何声学 ShoeBox 模型，用于验证多麦克风 TDOA / SRP-PHAT 定位流程。
模型未精确刻画金属笼条、鸡体遮挡、鸡体散射、衍射、复杂鸡舍设备噪声以及独立 USB 麦克风之间的时钟漂移。
因此，本实验结果主要反映算法在受控仿真条件下的定位性能，不能直接等同于真实鸡舍部署效果。
本实验输出的 source_id 为场景内临时声源编号，不对应固定蛋鸡个体，也不建立个体声纹。

## 7. 输出文件

- 单源逐场景结果：`results/week2/day3/single_source.csv`
- 定位结果表：`results/week2/day3/localization_results.csv`
- E1 汇总统计表：`results/week2/day3/experiment_summary.csv`
- 固定声源位置表：`results/week2/day3/single_source_positions.csv`
- 误差 CDF 图：`results/week2/day3/single_source_error_cdf.png`
- 空间误差图：`results/week2/day3/single_source_spatial_error.png`
- 场景配置：`results/week2/day3/e1_single_source_scene_config.yaml`
- 实验报告：`results/week2/day3/e1_single_source_report.md`
