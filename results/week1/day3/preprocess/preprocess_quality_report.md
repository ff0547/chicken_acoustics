# 第 1 周周三：音频清洗与标准化质量报告

## 1. 处理目标

- 输入目录：`data/raw_sources`
- 清洗音频输出目录：`data/processed_sources`
- 切分片段输出目录：`data/processed_segments`
- 目标采样率：`48 kHz`
- 输出声道：`单声道`
- 输出格式：`.wav`
- 切分长度：`1～5 s`
- 处理流程：转单声道、重采样、去直流、静音裁剪、峰值归一化、切分、可视化

## 2. 总体统计

- 扫描原始音频文件数：449
- 成功清洗完整音频文件数：443
- 切分片段总数：29041
- 有效片段数：29041
- 完整音频削波样本总数：0
- 片段削波样本总数：0

结论：达到今日验收标准，满足 `≥80 段有效片段、无削波、格式统一`。

## 3. 数据集统计

| dataset | total_files | processed_files | valid_segments |
| --- | --- | --- | --- |
| mendeley_poultry_vocalization | 346 | 346 | 2383 |
| zenodo_laying_hens_stress | 103 | 97 | 26658 |

## 4. 文件处理状态统计

| reason | count |
| --- | --- |
| ok | 425 |
| no_valid_segment_1_to_5_sec | 18 |
| error: MemoryError: Unable to allocate 2.65 GiB for an array with shape (2048, 347134) and data type float32 | 1 |
| error: MemoryError: Unable to allocate 2.67 GiB for an array with shape (2048, 349607) and data type float32 | 1 |
| error: MemoryError: Unable to allocate 2.65 GiB for an array with shape (2048, 346960) and data type float32 | 1 |
| error: MemoryError: Unable to allocate 2.65 GiB for an array with shape (2048, 346712) and data type float32 | 1 |
| error: MemoryError: Unable to allocate 2.70 GiB for an array with shape (2048, 354321) and data type float32 | 1 |
| error: MemoryError: Unable to allocate 2.67 GiB for an array with shape (2048, 350077) and data type float32 | 1 |

## 5. 样例图输出

- `results\week1\day3\preprocess\examples\sample_01_waveform.png`
- `results\week1\day3\preprocess\examples\sample_01_spectrum.png`
- `results\week1\day3\preprocess\examples\sample_01_logmel.png`
- `results\week1\day3\preprocess\examples\sample_02_waveform.png`
- `results\week1\day3\preprocess\examples\sample_02_spectrum.png`
- `results\week1\day3\preprocess\examples\sample_02_logmel.png`
- `results\week1\day3\preprocess\examples\sample_03_waveform.png`
- `results\week1\day3\preprocess\examples\sample_03_spectrum.png`
- `results\week1\day3\preprocess\examples\sample_03_logmel.png`

## 6. 输出文件

- 文件级质量报告：`results\week1\day3\preprocess\preprocess_file_report.csv`
- 片段清单：`results\week1\day3\preprocess\segment_manifest.csv`
- Markdown 质量报告：`results\week1\day3\preprocess\preprocess_quality_report.md`
