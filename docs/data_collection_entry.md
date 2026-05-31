# SE-Bench Research Data Collection Entry

This file mirrors the columns in `SE-Bench Research数据收集.xlsx` for the battery task. It intentionally excludes API keys, passwords, SSH details, and private credentials.

| Column | Value |
|---|---|
| 进度 | 题包/数据/评测/本地审计/远端 live-auth preflight 已完成；当前新包待同步到 VM 后重跑 30m/1h/2h/8h |
| 任务名称 | 面向储能电池寿命评估的 SOH/RUL 预测与异常循环识别 |
| 任务分类 | 工程/工科 |
| VLM/LLM执行 | LLM |
| 是否允许联网 | 否 |
| 是否需要GPU资源 | 否 |
| 所需软件/环境 | Python 3.11；numpy==1.26.4；pandas==2.2.2；scikit-learn==1.4.2；scipy==1.13.0；joblib==1.4.2；SE-Bench harness；Docker |
| 所需资源 | CPU 8 cores；内存 16 GB；磁盘 20 GB；无 GPU |
| 确认是否Linux系统可跑通 | 是。远端 Linux VM 已完成资产校验、Docker work/judge 镜像构建和直接 judge scorer smoke |
| 任务描述（题干，agent prompt） | Agent 需要改进 CPU-only Python 电池健康建模流水线，在可见 train/dev 数据上训练，并对隐藏 eval cell-cycle 特征预测 SOH、RUL、异常类型和异常严重度。输出列为 `cell_id, cycle_index, predicted_soh, predicted_rul_cycles, predicted_anomaly_type, predicted_anomaly_severity`。评分奖励 SOH/RUL 精度、EOL/knee-cycle 校准、异常检测、异常类型分类和严重度排序。 |
| 人类最优/本人最优完成度 | 参考专家目标为 100 分；当前普通 HGB 参考实现为 10.652701，starter 为 3.520740，本题保留 30+ 分给更深入的异常专门建模、per-chemistry/protocol 建模、knee/RUL calibration 和集成搜索 |
| 人类最优效果/本人完成 | `baseline/reference_hgb.py`、`baseline/reference-hgb-score-log.json`、`baseline/reference_hgb_submission.csv` |
| 真实性说明 | 电池 SOH/RUL、knee point、EOL 和异常循环识别是储能验证实验室和电池 fleet 运维中的真实任务；题目包含混合工况、化学体系差异、温度/阻抗/容量退化、sensor drift、thermal event、resistance spike、capacity drop 和 recovery relaxation |
| 工作量说明 | 2-3h schema/leakage/distribution audit；3-5h SOH/RUL baseline；3-5h knee/EOL calibration；4-6h anomaly detection/type modeling；3-5h feature/threshold/model tuning；2-4h packaging/reproducibility/hidden-style validation，总计 20h+ 有效工作 |
| 人类工作时长/h | 20+ |
| 任务输入 | `agent-start/`，包括 `train.py`、`predict.py`、`src/`、`requirements.txt`、可见 `dev-data/train/`、`dev_features.csv`、`dev_labels.csv` 和 `sample_submission.csv` |
| 评分方案 | 0-100 连续分。权重：SOH 0.24，RUL 0.24，EOL 0.10，knee 0.10，anomaly F1 0.18，anomaly type 0.09，severity 0.05；锚点压制 ordinary HGB 到 15 分以下，2h 验收上界 30 分 |
| 评分文件/脚本 | `scorer/evaluate.py`、`scorer/run_eval.py`、`scorer/score.sh` |
| key是否下发 | VM live-auth preflight 已通过；文档不记录、不展示 key |
| 结构化JSON数据 | `task.yaml`、`harness/battery_soh_rul_anomaly.json`、`docs/submission_manifest.json`、`docs/remote_preflight.json` |
| 环境空跑说明 | scorer 和 baseline smoke 为分钟内完成，评测等待占比预期低于 50%；真实长跑时每 300s auto-eval 一次，agent 可持续修改代码 |
| 附件（说明文档） | `docs/data_audit.md`、`docs/workload_and_authenticity.md`、`docs/pollution_control.md`、`docs/pre_submission_review.md`、`docs/environment_setup_cn.md`、`docs/design_and_validation_plan.md`、`docs/submission_manifest.md` |
| 源链接（GitHub/文档） | 本地题包路径：`Xpert/SE-Bench Research/battery-soh-rul-benchmark`；资产路径：`Xpert/SE-Bench Research/harness-assets/battery_soh_rul` |
| 工作目录cwd | `/home/workspace/battery-soh-rul-benchmark/agent-start` |
| spec存放目录 | 无独立 specs 目录；任务说明在 harness `agent_query` 和 `agent-start/README.md` |
| 代码仓 | `battery-soh-rul-benchmark` |
| 截图1 - 模型执行30min分数 | 待当前新包同步到 VM 后运行；完成后用 `tools/collect_long_run_evidence.py` 汇总 |
| 截图2 - 模型执行2h分数 | 待 30m/1h 满足阈值后运行；完成后写入 `docs/long_run_evidence.json` |
| 截图3 - 模型执行8h以上分数 | 待 30m/1h/2h 满足阈值后运行；完成后写入 `docs/long_run_evidence.json` |
| 模型执行的不同轮次结果 | 旧包远端 no-op/HGB smoke 和中止的 1h run 因资产过期作废；当前新包本地分数 starter `3.520740`、HGB `10.652701`，待重跑远端长验证 |
