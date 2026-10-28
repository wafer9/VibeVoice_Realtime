# 仓库执行指引

## 训练计划

本仓库以 [README.md](README.md) 开头的模型训练计划为持续执行基准。进行模型、数据、训练、推理或评估工作前先阅读该计划，并检查当前阶段和既有实验结果。

用户后续明确指示优先。用户修改路线后同步更新 README；计划内的实现、修复、调参和小规模对照可自主推进，不要重复请求已经明确的授权。

## 代码骨架

按用户要求，以 `/nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg` 的骨架为基础：`run.sh`、`twinlakes.bin.train`、`Dataset/processor/padding`、`LMModel.from_audio_text_pretrained`、`Executor.train/cv`、`WarmupLR` 和 `.pt/.yaml` checkpoint。代码与测试统一使用本仓库的 `twinlakes/`，不另建独立训练框架或兼容包。用户已替换 twinlakes 源码；当前latent训练入口为 `conf/run_stage1_latent.yaml`（Qwen3-0.6B与VibeVoice原生DDPM训练head），`conf/run_stage1_fm.yaml` 仅保留为历史原型。清理时保留用户新版活跃模型计算路径；一阶声学MLP负分支forward已实现并通过小样本验证，FM和CFG推理仍待实现，不把旧原型验证算作当前验证。

## 固定方向

- 固定 TwinLakes 当前使用的配套 acoustic encoder 和 decoder，基础训练阶段均冻结。
- 训练流式自回归 LLM + 条件 DiT，DiT 使用噪声到数据的 linear flow matching。
- CFG负条件由上一帧声学latent经轻量MLP生成，首帧使用可学习audio_BOS；共享扩散head，不额外运行负分支LLM。2026-10-28用户要求forward同时计算两路loss，condition权重0.8、uncondition权重0.2，EOS loss沿用原逻辑；该forward已实现，CFG推理待接入。文本窗口不更新上一声学帧缓存，训练历史严格右移。
- 正分支必须接收真实 LLM hidden condition，不能误将全部正条件置零。
- 使用 Emilia 约 10 万小时数据，按 README 分阶段验收后扩展；基础模型合格后再做 SIM-RL。
- 主要目标是未见说话人的零样本克隆；不以固定音色上的改善替代该目标。

## 实施要求

- 保持 codec、音频归一化、latent 缩放、FM 目标、时间方向和采样器一致。
- 模型浮点参数、浮点 buffer 和传入模型的浮点张量统一使用配置 dtype；训练入口不强制 model.float()。索引、标签、长度和 bool mask 保留整数/布尔类型，loss 可显式用 FP32 计算。
- 不默认混用原 Realtime decoder 与 TwinLakes encoder，不假设 Realtime 权重直接适配新 latent 空间。
- 局部声学历史等未固定的建议作为对照实验记录，不冒充用户已确定的要求。
- 先检查 teacher forcing 位移、流式文本可见范围、因果掩码、EOS 与 loss mask，再扩展训练。
- 记录配置、数据版本、checkpoint 和实际评估；区分已实现、已验证、待实现和假设。
- 同步维护 README 的阶段状态。不要将只写好代码或通过小样本检查描述为已完成大规模训练。
