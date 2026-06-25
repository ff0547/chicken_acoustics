# Day2 数据来源整理记录

## 1. 任务目标

本日任务为下载至少两个公开鸡叫/家禽声学资源，建立 `source_manifest.csv`，记录每个音频文件的来源 URL、许可证、下载日期、原始采样率、时长和标签，并剔除许可证不清的数据。

## 2. 正式采用的数据源

### 2.1 Poultry Vocalization Signal Dataset for Early Disease Detection

- 来源：Mendeley Data
- 许可证：CC BY 4.0
- 数据内容：Healthy、Noise、Unhealthy 三类家禽声音文件
- 使用状态：已纳入 `source_manifest.csv`

### 2.2 Vocalization Patterns in Laying Hens - An Analysis of Stress-Induced Audio Responses

- 来源：Zenodo
- 许可证：CC BY 4.0
- 数据内容：control / treatment laying hen vocalization data
- 使用状态：已纳入 `source_manifest.csv`

## 3. 候选但暂不纳入的数据源

### ChickenLanguageDataset

- 来源：GitHub
- 当前状态：已下载并保留在 `data/candidate_sources/`
- 剔除原因：当前未确认明确数据许可证
- 处理方式：生成 `candidate_manifest_unclear_license.csv`，但不纳入正式 `source_manifest.csv`

## 4. 暂缓下载的大规模数据源

### ChickenSense

- 来源：Zenodo
- 许可证：CC BY 4.0
- 当前状态：暂不下载
- 原因：全量 `Dataset.zip` 约 68.9GB，下载、解压和预处理成本较高；当前 Day2 阶段不需要全量真实长时数据。
- 后续用途：真实长时鸡舍音频验证、真实背景噪声建模、模型泛化测试。

## 5. 输出文件

- `data/manifests/source_manifest.csv`
- `data/manifests/source_summary.csv`
- `data/manifests/candidate_manifest_unclear_license.csv`
- `data/manifests/rejected_sources.csv`
- `data/manifests/deferred_large_sources.csv`

## 6. 后续处理

下一步将根据 `source_manifest.csv` 对正式数据进行统一采样率转换、响度归一化、静音裁剪、有效片段筛选，并生成可用于 Pyroomacoustics 仿真的干净声源片段。