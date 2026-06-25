# 第 1 周第 5 天：单源和多源仿真器质量报告

## 1. 当前任务定位

本阶段属于第 1 周仿真器功能验证，不是论文级 E1～E10 正式实验。
当前生成 20 个测试场景，用于验证 RIR、source image、mixed audio、随机种子复现和求和一致性。
同时，本脚本已经在 YAML 和 CSV 中写入后续论文级实验所需的元数据字段。

## 2. 功能目标

- 使用 Pyroomacoustics 实现 RIR 生成
- 保存 image-source 模型几何信息
- 生成单源与多源混合音频
- 支持声源起始时间、增益、RT60、SNR 和随机种子
- 生成 20 个可复现测试场景
- 验证 clean mix 与 source image 求和一致
- 验证 noisy mix 与 source image + noise 求和一致
- 预留 experiment_id、group_id、ground truth 坐标和 planned_metrics 字段

## 3. 总体统计

- 场景数量：`20`
- 单源场景数量：`7`
- 双源场景数量：`13`
- 多源场景数量：`13`
- 声源总数：`33`
- clean mix 求和验证通过场景数：`20`
- noisy mix 求和验证通过场景数：`20`
- clean mix 最大误差：`1.19209290e-07`
- noisy mix 最大误差：`1.38534233e-07`

结论：今日验收通过，20 个场景均可复现，混合信号与 source image 求和关系一致。

## 4. 后续论文级实验衔接
- 后续论文级实验将完整覆盖 E1～E10，并输出 localization_results.csv、separation_results.csv、experiment_summary.csv 和 daily_log.csv。

当前 20 个场景不作为论文级统计实验结果。后续正式实验将由 `06_build_paper_sim_scenes.py` 扩展。

后续建议：

- E1：麦克风数量 4/6/8/12，每组 100 场景，计算 MAE、P90、10 cm 命中率
- E3：SNR 0/5/10/20/30 dB，每组 100 场景，计算误差和漏检率
- E5：RT60 0.1/0.3/0.5/0.7 s，每组 100 场景，计算误差和峰值比
- E6：单通道 / DAS / MVDR / LCMV，每组 100 场景，计算 SI-SDRi 和 Log-Mel 距离

## 5. 输出文件

- 场景配置：`configs\week1\day5\sim_scenes.yaml`
- 场景清单：`results\week1\day5\simulate_scene\scene_manifest.csv`
- 声源清单：`results\week1\day5\simulate_scene\source_manifest.csv`
- 验证结果：`results\week1\day5\simulate_scene\simulation_validation.csv`
- 仿真数据目录：`data\simulated_scenes\week1\day5`
