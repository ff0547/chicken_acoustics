# 第 2 周周二：二维近场 SRP-PHAT 声源定位报告

## 1. 当前任务

- 建立 z=0 平面二维网格。
- 预计算各网格点到各麦克风对的 TDOA 索引。
- 生成 SRP-PHAT 热力图。
- 验证无噪声单源场景下，SRP-PHAT 峰值是否接近真实声源位置。
- 验收标准：无噪声单源二维定位误差 ≤ 5 cm。

## 2. 基准场景

- 鸡笼尺寸：`1.2 × 0.75 × 0.6 m`
- 采样率：`48000 Hz`
- 声速：`343.0 m/s`
- 麦克风布局：`mic_8`
- 麦克风数量：`8`
- 麦克风对数量：`28`
- 搜索平面：`z = 0.0 m`
- 网格间距：`0.02 m`
- 网格大小：`59 × 36`
- 声源真实位置：`[0.420, 0.315, 0.000] m`
- 场景类型：无混响、无噪声、单源、直达声

## 3. 定位结果

- SRP-PHAT 峰值位置：`[0.420, 0.320, 0.000] m`
- 二维定位误差：`0.500 cm`
- 验收阈值：`≤ 5.0 cm`
- 验收结论：`通过`

## 4. 方法说明

对每个候选网格点 p，先计算该点到每一对麦克风的理论 TDOA：

```text
tau_ij(p) = [distance(p, Mi) - distance(p, Mj)] / speed_of_sound
```

然后把 tau_ij(p) 转成 GCC-PHAT 曲线下标，在对应位置读取相关值并累加：

```text
score(p) = sum_ij GCC_PHAT_ij(tau_ij(p))
```

最后取总分最高的网格点作为估计声源位置。

## 5. 输出文件

- SRP 热力图：`results/week2/day2/12_nearfield_srp_2d/srp_heatmap.png`
- 网格得分表：`results/week2/day2/12_nearfield_srp_2d/srp_grid_result.csv`
- 峰值结果表：`results/week2/day2/12_nearfield_srp_2d/srp_peak_result.csv`
- 麦克风对 GCC 峰值表：`results/week2/day2/12_nearfield_srp_2d/gcc_pair_peak_table.csv`
- TDOA 索引缓存：`results/week2/day2/12_nearfield_srp_2d/tdoa_index_cache.npz`
- 场景配置：`results/week2/day2/12_nearfield_srp_2d/nearfield_srp_scene.yaml`
- 报告：`results/week2/day2/12_nearfield_srp_2d/nearfield_srp_2d_report.md`
