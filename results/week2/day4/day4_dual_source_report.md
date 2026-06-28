# 第 2 周 Day4：双源多峰检测实验报告

## 1. 实验目的

本实验在 Day3 已通过验收的二维近场 SRP-PHAT 单源定位基础上，扩展到双源场景。
本版采用短时帧聚合 SRP-PHAT 与聚类先验增强方法，最终在聚合得分图上执行局部极大值检测和 NMS，以满足 Day4 多峰检测要求。

## 2. 方法流程

1. 生成两个空间位置不同的声源。
2. 两个声源在同一场景中以短时事件形式发声。
3. 使用 Pyroomacoustics 生成混响加噪多通道信号。
4. 根据短时能量选择有效帧。
5. 对每个有效帧复用 Day3 的单源 SRP-PHAT 定位。
6. 由帧级定位点构造空间密度图。
7. 将多帧定位点进行加权聚类，生成聚类先验图。
8. 融合密度图和聚类先验图，得到 hybrid_map。
9. 在 hybrid_map 上执行局部极大值检测。
10. 使用 NMS 去除重复峰，并输出两个预测峰。
11. 使用匈牙利算法匹配预测峰和真实源。
12. 统计双源命中率、漏检率和虚警率。

## 3. 基准参数

- 麦克风布局：`mic_8`
- 麦克风数：`8`
- 活动源数：`2`
- 源间距：`>= 30 cm`
- RT60：`0.30 s`
- SNR：`20.0 dB`
- 网格间距：`2.0 cm`
- 命中半径：`10 cm`
- 场景数：`100`

## 4. 统计结果

| experiment_id | variable | value | n | num_mics | dual_hit_rate | miss_rate | false_alarm_rate | mean_error_cm | std_error_cm | median_error_cm | p90_error_cm | ci95_dual_hit_rate | pass_day4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4_day4_baseline | source_distance | >=30cm | 100 | 8 | 0.9300 | 0.0350 | 0.0350 | 4.2250 | 2.6464 | 3.7608 | 7.5112 | 0.0500 | True |

## 5. 验收结论

- 验收标准：`源间距 >= 30 cm 时，双源命中率 >= 0.80`
- 当前双源命中率：`0.9300`
- 是否通过：`True`

## 6. 输出文件

- 逐场景结果：`results/week2/day4/dual_source.csv`
- 汇总结果：`results/week2/day4/dual_source_summary.csv`
- 示例聚合得分图：`results/week2/day4/example_srp_peaks.png`
- 示例帧定位图：`results/week2/day4/example_frame_positions.png`
- 失败样本图：`results/week2/day4/failure_cases/`
- 配置文件：`results/week2/day4/day4_dual_source_config.yaml`

## 7. 运行时间

- 总耗时：`160.95 s`
