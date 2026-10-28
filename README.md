# 模型训练计划（后续工作的执行基准）

更新日期：2026-10-28。

**本项目按照以下计划推进：固定 TwinLakes 配套 acoustic codec，训练支持参考音频零样本克隆的流式 LLM + DiT；DiT 使用 flow matching，CFG 负条件由上一帧声学 latent 经轻量 MLP 得到；使用 Emilia 约 10 万小时数据分阶段训练，基础模型稳定后再进行 SIM 强化学习。**

**当前实现状态：用户已整体替换 `twinlakes/`，本次在这版代码上完成原生 PyTorch 清理。早期 FM 原型与19项测试的记录属于历史；当前代码是否满足本计划，以第5节列出的验证和缺口为准。**

后续修改模型、准备数据、运行训练和评估前，先阅读本计划。用户后续的明确指示优先；用户调整路线后，应同步更新本文件。计划范围内的调参、修复和小规模对照实验可以直接推进，不必反复请求确认。实验结果不得写成已经确认的结论。

## 1. 已确定的核心方案

| 项目 | 执行方案 |
| --- | --- |
| 主要目标 | 提升任意新说话人的零样本音色克隆能力，同时维持文本正确性、自然度和流式生成能力 |
| 代码骨架 | 沿用 TwinLakes_cfg 的 twinlakes/bin、dataset、models、utils、Executor 和 WarmupLR；不维护另一套独立训练框架 |
| Acoustic codec | 使用 TwinLakes 当前使用的配套 encoder 和 decoder；基础训练阶段均冻结 |
| 生成架构 | 自回归 LLM 提供条件，轻量 DiT 生成下一个 acoustic latent |
| DiT 训练目标 | 采用线性路径 conditional flow matching |
| CFG 负条件 | 上一帧声学latent经轻量MLP生成负条件，首帧使用可学习audio_BOS；forward已同时计算正/负loss，权重0.8/0.2；CFG推理接入待完成 |
| 负分支 LLM | 首版不运行额外的负分支 LLM，不复制原 Realtime 的负分支自回归路径 |
| 训练数据 | Emilia 约 10 万小时；以实际可访问、清洗后的 manifest 统计为准 |
| 推进顺序 | 协议与数据检查 → 小样本跑通 → 数百小时验证 → 数千小时扩展 → 全量训练 → SIM-RL |
| 后续 RL | 借鉴 FlowTTS-GRPO，联合考虑 SIM、ASR 和音质，并验证未见说话人的收益 |

本方案包含基础 TTS 的有监督训练阶段。更换 codec latent、FM 目标和负条件机制后，不假设原 Realtime checkpoint 可以直接兼容。

## 2. 模型与损失

### 配套 codec

TwinLakes 源码参考位置：

```text
/nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg
```

其当前配置使用的完整 codec 权重：

```text
/nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B
```

音频统一为 24 kHz 单声道，使用一致的幅度归一化。codec latent 为 64 维，压缩率 3200，帧率为 7.5 Hz。离线编码首版保存 encoder mean，posterior sampling 是否作为数据增强另做对照。

必须区分 raw codec latent 和生成模型使用的归一化 latent：

```text
z_model = (z_raw + bias) * scale
z_raw   = z_model / scale - bias
```

首版沿用本地 VibeVoice-1.5B 权重中的参数：

```text
scale = 0.1962890625
bias  = -0.04931640625
```

数据缓存必须记录 codec 权重来源、encoder mean/sample 模式、音频归一化、latent 是否已缩放以及 scale/bias，防止重复缩放和混用版本。不要使用原 Realtime 的缩放参数解释这套 codec 的模型空间。

### LLM + DiT

参考音频经 encoder 得到的 latent 作为音色上下文。对于第 i 帧，LLM 条件 h_i 只能依赖参考上下文、当前允许看到的目标文本及目标语音历史 z_<i，不能读取目标帧 z_i 或未来声学帧。

```text
参考音频 → 冻结 encoder → 参考 latent ─┐
当前可见文本 + 已生成/teacher-forced latent ─┤
                                          ↓
                                     自回归 LLM
                                          ↓ h_i
Gaussian noise → 条件 DiT 的 FM 采样 → z_i
                                          ↓
                              反归一化 → 冻结 decoder → 音频
```

LLM 使用预训练权重初始化并参与 TTS 训练。早期 FM 原型使用 `Qwen3-0.6B-Base`（LLM 596,049,920 参数、DiT 7,838,272 参数、总训练参数 605,012,545），配置保留于 [conf/run_stage1_fm.yaml](conf/run_stage1_fm.yaml)，这些数字不代表当前新版。当前实际 [latent训练配置](conf/run_stage1_latent.yaml) 使用完整28层 Qwen3-0.6B-Base，hidden_size=1024；factory 重新初始化 VibeVoice 原生4层扩散head并将hidden_size设为1024，head共54,923,264参数，未使用单独的diffusion_head_proj。LLM为596,049,920参数；加入一阶声学MLP负分支及audio_BOS后，总可训练参数为652,372,290（新增279,872）。Acoustic connector 与扩散head重新初始化，不默认沿用 Realtime 权重。当前训练目标仍是DDPM v-prediction，linear FM尚未接入。

原先 FM 原型使用固定的“每次读取最多 5 个文本 token，再生成 6 个声学帧”的可见范围调度；当前用户新版输入完整目标文本，该流式调度尚未迁移。该调度不是 forced alignment；文字无法在目标音频结束前完全暴露的样本直接过滤，不截断目标音频或转写来凑长度。其对真实语音的对齐和流式质量仍需 P1/P2 验证。

线性 FM 的定义统一为噪声到数据：

```text
epsilon ~ N(0, I)
t ~ Uniform(0, 1)                 # 首版基线
x_t = (1 - t) * epsilon + t * z_i
u_target = z_i - epsilon
L_FM = masked_mean(||DiT(x_t, t, h_i) - u_target||^2)
```

采样从 t=0 的噪声积分到 t=1 的模型空间 latent。流速目标、时间方向和采样器必须保持一致，不能直接将 DDPM 的 v-prediction 目标或 DPM-Solver 更新式套到线性 FM 上。

联合训练 EOS/停止预测，避免无限生成、提前停止和尾部重复。总损失至少包含有效目标帧上的 FM loss 和 EOS loss；参考音频、padding、非目标位置应正确屏蔽。记录各项损失及其权重，不能只观察总 loss。

### 一阶声学先验与 CFG（forward已实现，推理待接入）

2026-10-28用户确认并要求实现：用上一帧声学latent经轻量MLP生成负条件，替代原先的zero LLM condition方案。当前只修改模型构造与训练forward；新增MLP为64→256→LLM hidden_size，中间SiLU，audio_BOS为可学习64维向量（初始化为零）。MLP输出head条件，不直接预测一个确定的下一帧latent做CFG相减。每次forward运行一次LLM，两路共享扩散head并使用相同噪声、时间步和DDPM v目标，按0.8/0.2加权，而非随机丢弃条件。

```text
h_pos_i = LLM(参考音频, 当前允许看到的文本, z_<i)
z_prev = z_(i-1)                 # 第一帧为可学习 audio_BOS
h_neg_i = MLP(z_prev)
v_pos = head(x_t, t, h_pos_i)
v_neg = head(x_t, t, h_neg_i)
condition_loss = MSE(v_pos, v_target)
uncondition_loss = MSE(v_neg, v_target)
diffusion_loss = 0.8 * condition_loss + 0.2 * uncondition_loss
loss = loss_text + diffusion_loss  # loss_text是现有EOS分类loss
```

正分支使用真实LLM hidden，保留回到LLM/connector的梯度；负分支训练MLP、audio_BOS与共享head。两路MSE在有效目标帧上用FP32计算；padding和参考帧不参与，空目标时用零loss保持新增参数在DDP计算图中。Executor自动记录condition_loss、uncondition_loss及加权diffusion_loss。该负分支表示p(z_i | z_(i-1))的一阶声学先验，仍包含历史声音的信息，并非对所有信息完全无条件。CFG强度待对照确定。

本次40项测试通过：包含原始音频/缓存latent在FP32/BF16/FP16下的前向反向、逐样本右移和BOS、padding排除、单次LLM调用、两路相同噪声/时间步、加权公式及梯度隔离、空目标分支。同步更新了仍假设存在diffusion_head_proj的旧测试，沿用当前无该投影的模型。见 [测试日志](outputs/uncondition_training_20261028/tests.log)。未启动正式训练；现有generate_cfg仍待切换到该MLP负条件及声学帧缓存，不能将forward验收算作CFG推理验收。

后续训练观察（2026-10-28约17:11）：用户已启动双分支正式训练，`exp/s0/log/0.log`到step=6000；不代表完成epoch或语音质量验收。实际保存配置为`sort: false`。172当前`/data/joe/data/Emilia_latent_shards/train/metadata.json`确认27,806,940对、2,781包均已按seed=20261028全局随机且每包混合EN/ZH，输出目录由用户使用train而非此前建议的train_shuffled。当前每100步日志仅记录rank 0当次更新的2个micro-batch均值（每卡batch=16），并非最近100步或全卡平均；padding内argsort只改变同一batch的样本排列。60个日志点去趋势后未建立稳定周期，正负分支波动高度相关；批次难度和扩散噪声波动是待验证解释，尚无逐batch长度/语种/时间步统计，不能进一步确定原因。见 [诊断记录](outputs/loss_periodicity_20261028/report.json)。

Loss尺度与旧实验（用户补充）：用户报告此前未做平均的loss约64→36、lr=0.001时效果最好；旧实验的完整聚合公式、EOS权重、warmup和裁剪配置尚未逐项对齐，不将其认定为当前实验的等效对照。若旧式仅对64维求和且同样对有效帧平均，36对应当前逐元素MSE的0.5625；若旧式还对帧/batch求和，则不能仅除64换算。当前采用Adam、峰值lr=0.001、warmup=5000和grad_clip=1；TensorBoard在step=7900记录lr=0.0007955069，5000步后已记录梯度范数约0.1104～0.1904，均低于裁剪阈值，仅说明这些记录点未触发裁剪。当前condition_loss约0.64～0.67且继续下降，未验证达到旧实验音质。此次保持配置不变；需要对照的是loss归一化、EOS相对权重和裁剪行为，不能机械地按64倍缩放学习率。

推理使用同一份 DiT 参数：

```text
u_pos = DiT(x_t, t, h_i)
u_neg = DiT(x_t, t, MLP(z_prev))
u_cfg = u_neg + cfg_scale * (u_pos - u_neg)
```

cfg_scale=1 时只需计算正常条件。cfg_scale>1 时需要两种 DiT 条件预测，可以合批；不需要额外运行负分支 LLM。

维护每个样本独立的last_audio_latent：只在一帧完成去噪后更新，追加文本窗口时保持原值；新语音样本开始时重置为audio_BOS。这里取上一声学帧，不取LLM混合序列的上一位置。训练直接对目标声学序列右移：[BOS,z_0,z_1,...]对应[z_0,z_1,z_2,...]；禁止读当前帧或未来帧。训练和推理都使用经过同一scale/bias归一化的模型空间latent。额外声学历史窗口或独立说话人guidance仍属未确认的对照，不纳入首版。

## 3. 数据与零样本克隆协议

- 主训练使用同一说话人的不同句子构造参考 A 与目标 B，避免只学会相同录音的延续或复述。参考音频长度随机化，并在配置中记录分布。
- Emilia 的说话人标注和配对需检查，清理重叠语音、错误转写、低质量音频和疑似不同说话人的参考配对。
- 原版 Emilia 约 10.17 万小时，中文约 4.99 万小时、英文约 4.68 万小时，其余为其他语言。不能将“总计约 10 万小时”记作单一语言的规模。
- 训练、验证、测试按说话人及来源隔离；不能只随机切分句子。检查不同来源中同一说话人的泄漏风险。
- 固定独立的未见说话人评估集，并覆盖不同参考长度、录音条件和长句；跨语言克隆另列指标。
- 如果需要流式文本输入，训练时必须定义相应的文本可见范围、交错顺序或对齐方法。不能训练时看完整目标文本，推理时直接假设只看少量新文本也能保持原能力。
- 先定义正确的 teacher forcing 位移、因果 attention mask、EOS 标签和 loss mask，再进行大规模数据提取与训练。

### 当前执行的 Emilia 索引过滤（以 meta.py 任务为准）

用户最新任务见 [data/Emilia/meta.py](data/Emilia/meta.py) 开头说明。本次全量过滤按以下规则执行：

- 按目标音频 ID 关联元数据、SIM TSV 与 Qwen3-ASR WER TSV，严格要求 `SIM > 0.8` 且 `WER < 0.1`，两个等号边界均不保留。
- 英文 `text` 使用 WER TSV 第 4 列的原始 Qwen3-ASR 识别文本；中文 `text` 保留 Emilia 元数据原文。
- 新增 `prompt_wav`，按用户要求保存 SIM TSV 第 2 列的**音频 ID**；新增 `prompt_text`，英文使用该参考 ID 的 Qwen3-ASR 文本，中文使用该参考 ID 的 Emilia 原文。
- 分数门槛作用于目标，不额外要求参考 ID 通过 SIM/WER。无法找到参考文本的配对单独计数并排除，不使用目标文本代填。
- 其余原元数据字段保留。本次不额外加入先前建议的 DNSMOS、时长或去重质量门槛。输入文件不覆盖；输出按英文、中文顺序合并。

当前状态：已完成全量过滤，结果为 [train.jsonl](data/Emilia/train.jsonl)，统计为 [train.stats.json](data/Emilia/train.stats.json)，运行日志为 [train.filter.log](data/Emilia/train.filter.log)。输出大小 17,595,175,523 字节，约 16.39 GiB。完成前仅写 `.partial`；脚本拒绝覆盖已存在的输出。

| 语言 | 原始条数 | 保留条数 | 保留时长（小时） | 条数保留率 |
| --- | --- | --- | --- | --- |
| en | 18,136,271 | 12,739,860 | 35,377.03 | 70.25% |
| zh | 19,969,297 | 15,067,080 | 39,556.12 | 75.45% |
| 合计 | 38,105,568 | 27,806,940 | 74,933.15 | — |

时长按原元数据累加，没有解码全量音频。英文有 2,982 个、中文有 21 个通过 SIM 的目标缺少 WER；英文还有 132 个通过两个门槛的目标因参考缺少 Qwen3-ASR 文本而排除，中文参考文本无此缺失。所有通过两个门槛的目标均找到了目标元数据。

校验已通过：使用 [独立校验脚本](tools/verify_emilia_meta.py) 重新读取完整输出，确认分语言 SHA256 和条数与生成记录一致；从每种语言的 64 个等间隔字节窗口各取 8 行，共回查 1,024 条输出对应的原 SIM、WER、目标与参考文本，均匹配。结果见 [train.verify.json](data/Emilia/train.verify.json)。该校验不包含音频解码、真实说话人身份或划分泄漏验证。

```bash
python data/Emilia/meta.py
# 再次运行时指定新输出路径：
python data/Emilia/meta.py --output data/Emilia/train_v2.jsonl
```

此输出的 `prompt_wav` 为 ID，没有新增 `sim` 字段。当前原始音频读取器和打包脚本已支持按 Emilia ID 解析参考路径；原离线 `prepare_latent` 入口已被用户替换源码时移除，旧缓存命令不再适用。本次没有启动全量 latent 提取或真实模型训练。四项测试覆盖严格边界、乱序 ID 关联、两种语言的目标/参考文本来源、缺失分数、非有限分数及关联 ID 重复；参考未通过目标 WER 门槛仍可查到参考文本。

参考选择的待验证对照：本次继续使用用户提供的同说话人 top-1 SIM 配对，不自动改成前一个片段。针对任意新说话人克隆，建议后续比较 top-1、确认同人的相邻片段、同人合格候选池随机采样；SIM 作为配对质量检查，不默认越高越适合所有训练样本。候选池随机化仍需多个候选或原始声纹 embedding，当前 top-1 TSV 不足以直接实现。训练时按目标 SIM 选参考可作为监督数据构造；评估应预先固定参考，不能按待合成目标的真实音频挑选最优参考后宣称代表任意参考克隆能力。

相关工作采用不同训练协议：[VALL-E](https://arxiv.org/html/2301.02111v1) 的 AR 训练使用语音前缀上下文，NAR 从同一条语音随机取 3 秒参考；[F5-TTS](https://arxiv.org/html/2410.06885v1) 随机遮盖 70%–100% 语音帧进行补全；[MaskGCT](https://arxiv.org/html/2409.00750v1) 的 T2S 随机取语义序列前缀作参考。上述做法不等于必须使用数据集里的前一个独立切片；同句补全也必须正确划分条件区和预测区，不能直接给本项目完整目标的副本作为参考。候选池随机化是本项目的建议对照，尚未实施或证明更优。

### Emilia 配对清洗建议（先前抽样，非本次执行规则）

新的原始索引位于 `data/Emilia/`：`en.jsonl`、`zh.jsonl` 提供元数据，`en_sim.list`、`zh_sim.list` 为 `target_id prompt_id cosine_sim`。文件顺序不同，必须按 ID 关联两端元数据，不能按行拼接；当前 latent 准备入口需要包含 `prompt_wav` 的 paired JSONL，不能直接传这四个原始文件。

2026-10-27 使用 [审计脚本](tools/audit_emilia_pairs.py)，在每种语言的 SIM 文件上取 128 个等间隔字节窗口、每窗口 512 行，各抽到 65,536 对，再扫描完整元数据以关联目标和参考。以下是**抽样配对条数保留率，不是全量小时数或清洗完成率**。抽到的两端 ID 均关联成功；抽样中 DNSMOS 最低为 3.0。详情见 [审计报告](outputs/emilia_audit_20261027/report_v2.json)。

| 建议候选集 | SIM 下限 | 目标和参考 DNSMOS 下限 | EN 保留 | ZH 保留 |
| --- | --- | --- | --- | --- |
| 扩量基础池 | 0.80 | 3.0 | 82.5% | 84.7% |
| 首轮干净池 | 0.85 | 3.2 | 41.6% | 42.7% |
| 更严格对照 | 0.90 | 3.2 | 24.5% | 21.7% |

表中还要求不同 ID、相同说话人 ID、不同音频路径、非空目标文本、目标时长 1–20 秒、参考完整音频 3–30 秒和有效分数。样本目标本身均不短于 3 秒。表中尚未扣除不同 ID 的相同文本配对，也没有验证音频、转写、真实身份和跨来源泄漏。仅按 SIM≥0.95 会让 EN/ZH 分别只剩约 6.3%/3.2%，不建议作为主训练门槛；阈值只针对本批 SIM，需试听及独立声纹验证。

建议执行顺序：

1. 校验元数据 ID 唯一性、两端存在、语言/说话人一致，剔除 self-prompt、无效分数和坏音频；对目标与参考同时检查音质。抽样已发现 EN 921 对、ZH 419 对自己配自己的记录，须明确过滤。
2. 先从首轮干净池按语言、来源和说话人分层抽数百小时做 P1/P2，再扩到基础池；每阶段报告小时数和说话人覆盖，避免只保留少数容易获得高 SIM 的声音。20–30 秒目标单独留存，当前 20 秒上限是首版训练配置要求，不表示长音频质量差。
3. 相同文本配对单独记录并优先重配；排除 self-prompt 后，抽样中 EN 仍有 1,039 对、ZH 有 31 对不同 ID 但归一化文本相同的配对，不能只依赖 SIM 下限。高 SIM 只作为重复风险线索，不使用 SIM 上限直接判重。必要时检查音频指纹/重叠、ASR 一致性、重叠人声及静音。按语言检查异常语速，避免用一个字符率阈值混筛中英文。
4. 划分 train/validation/test 后在各自划分内配对，参考也不得跨划分。当前按 `language_B` 来源划分，跨来源同一真人仍需独立声纹聚类/抽检，不能宣称严格未见说话人。
5. 首版继续记录 2–5 秒参考裁切；现有 SIM 来自完整参考，抽检裁切后的有效语音与 SIM。后续可在通过质量和同人检查的 top-K 参考中随机采样，提高参考条件多样性；现有 top-1 文件不足以直接实现该项。

Emilia 官方 [过滤实现](https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/utils/tool.py) 默认包含 DNSMOS≥3、3–30 秒和字符时长异常过滤，但这不替代本项目的配对与零样本隔离检查。本节记录的 DNSMOS 等门槛仍属建议，尚未全量执行；当前实际过滤以之前一节的用户最新任务为准。

固定 codec 后优先离线提取 latent。按 10 万小时、7.5 Hz、64 维、BF16 估算，latent 本体约 346 GB，不含索引、padding 和其他字段。实际容量、清洗后小时数与训练吞吐需测量后记录。

## 4. 分阶段执行与验收

### P0：固定训练协议与可复现配置

- [ ] 确认全量数据路径、语言配比、可用 GPU、显存与输出目录。
- [x] 固定 codec 版本、预处理、latent 定义与缓存元数据。
- [ ] 在当前用户新版上重新确认 LLM/DiT 配置、流式文本可见规则与说话人划分。
- [ ] 在当前用户新版上对齐linear FM、一阶声学MLP负分支CFG与训练/推理历史；双分支forward已实现并通过小样本测试，FM与CFG推理尚未接入，旧原型记录不作为当前验收。

验收：小批数据可完整前向、反向、保存和恢复；梯度有限，正常条件输入不为全零，训练目标和 mask 无信息泄漏。

### P1：小样本端到端跑通

- [ ] 用小规模、可人工核对的多说话人样本验证过拟合与文本-音频对应关系。
- [ ] 验证参考音频编码 → LLM 条件 → FM 采样 → 原配 decoder 的完整路径。
- [ ] 检查正常/一阶声学负条件均获得有效训练，CFG公式、audio_BOS、上一声学帧缓存与采样方向正确。
- [ ] 验证 EOS、逐帧与流式解码、checkpoint 恢复、原始样本与 latent 缓存的一致性。

验收：能够生成可辨识、对应目标文本的语音，能够正确结束；更换参考音频对音色有可验证影响。

### P2：数百小时验证架构

- [ ] 训练多说话人基线，建立独立验证集的 SIM、WER/CER、音质和时延曲线。
- [ ] 对照 cfg_scale=1 与使用 CFG，检查 condition dropout 与采样步数。
- [ ] 对照一阶声学先验CFG与cfg_scale=1，评估未见说话人SIM、WER与连续性；额外历史窗口另列实验。
- [ ] 评估未见说话人克隆、长句音色稳定性和流式文本输入。
- [ ] 实测数据加载、LLM、DiT 和音频解码的吞吐与显存。

验收：未见说话人的音色条件有效，语义与停止行为稳定；架构问题和吞吐瓶颈已有明确处理。

### P3：数千小时扩展

- [ ] 扩大说话人与语言覆盖，验证数据规模增加带来的收益。
- [ ] 完成分布式训练、断点续训、数据分片与评估自动化。
- [ ] 检查真实生成中的累积误差、重复、漏读、音色漂移与尾部异常。

验收：模型质量与训练稳定性达到预先记录的门槛，且有实测吞吐用于估算全量资源。具体门槛在 P2 形成基线后写入，不能事后修改标准掩盖退化。

### P4：Emilia 全量训练

- [ ] 按实际清洗后规模训练约 10 万小时数据，持续保存配置、manifest 版本、权重和固定评估结果。
- [ ] 定期报告实际见过的小时数、声学帧数、更新步数和各语言指标。
- [ ] 依据独立验证集选择 checkpoint，不仅按训练 loss 或最后一步选择。

验收：完整的零样本克隆、文本正确性、音质、长音频稳定性及流式性能评估；保留可复现的基础模型。

### P5：SIM 强化学习

- [ ] 从验证合格的基础模型出发，先尝试冻结 codec 与 LLM，仅训练 FM head。
- [ ] 适配 linear FM 的随机采样、轨迹记录、log-prob 与组内优势；记录 CFG 设置。
- [ ] SIM 为主要奖励，同时加入 ASR 和音质约束，并约束模型偏离基础模型的程度。
- [ ] 使用独立声纹模型、未见说话人和人工试听验证收益，检查奖励投机。
- [ ] 仅在 head-only 瓶颈有证据时，进一步评估联合训练 LLM。

验收：相对固定的基础模型，独立声纹指标与试听支持 SIM 改善，WER/CER、音质和流式性能没有不可接受的退化。

## 5. 当前状态与已验证事实

本次基于用户整体替换后的源码清理，未恢复旧模型或旧训练框架。当前仅支持原生 PyTorch 单进程与 DDP；已移除 DeepSpeed 初始化、ZeRO 估算/转换、专用配置、训练与 checkpoint 分支，以及专用同步进程组和钉钉通知。清理前源码及旧测试保存在 [备份](outputs/twinlakes_cleanup_20261027/before.tar.gz)。

当前 `twinlakes/` 从 18 个 Python 文件、5,472 行精简为 13 个文件、1,941 行；清理阶段保留的5个模型方法经 AST 哈希比较，计算代码均与清理前相同；之后按用户核对修正了 factory 的 VibeVoice 条件投影维度，详见 [投影检查](outputs/vibevoice_projection_20261027/report.json)。删除未使用的 compile/autocast/sampling/file_utils/mask 模块、多余调度器、无调用的 DataPipe 和旧模型推理分支。详情见 [清理报告](outputs/twinlakes_cleanup_20261027/report.json)。

新增 [Emilia latent 提取入口](tools/extract_emilia_latent.py)：按用户要求遍历原始 `en.jsonl`、`zh.jsonl` 的全部 MP3，GPU 编码 acoustic tokenizer mean，每条音频对应一个 `.latent` 文件。仅允许 GPU，CUDA 不可用时直接失败。路径映射、分 rank 不重不漏、续跑与坏缓存校验、原子写入失败和 codec 协议冲突的4项测试通过；实际343,696,032参数 encoder 的权重 strict load 和来源指纹检查通过。2026-10-27 用户调整 CLI 权限后，GPU 4 的实际 BF16 编码验收通过：英文、中文各2条完整音频共31.151秒，输出236帧，全部有限；与原生 `VibeVoiceAcousticTokenizerModel.encode(...).mean` 逐元素完全一致，最大误差为0。见 [GPU 验证](outputs/emilia_latent_20261027/gpu_validation.json) 和 [权重检查](outputs/emilia_latent_20261027/encoder_state_validation.json)。此前 [GPU 不可见记录](outputs/emilia_latent_20261027/report.json) 为权限调整前的历史状态。

2026-10-27 用户进一步指定输出先按大写 `EN/`、`ZH/` 分语言，脚本、测试和文档已同步。旧输出目录被外部移除后，本会话已按新布局在 GPU 4 重新生成中英文各2条，均与此前经原生 encoder 验证的 GPU 基准逐元素完全相同，见 [当前目录验证](outputs/emilia_latent_live_verify_20261027/report.json)。本会话曾启动的单卡全量任务已按用户要求停止，保留已有特征；状态见 [启动记录](outputs/emilia_latent_20261027/full_launch.json)。全量提取由用户在其他机器自行启动，本会话只做验证和有明确样本上限的性能测试，不自行重启全量。此前 root 用户8-rank任务导致权限冲突的情况保留为 [历史观察](outputs/emilia_latent_20261027/concurrent_run_observation.json)。此次数据编码不代表大模型训练或全量编码质量验收已经完成。

已完成的检查：

- 当前19项测试通过（含 VibeVoice 条件投影、配置 dtype、真实小模型前向/反向与 checkpoint 类型恢复），覆盖原生训练入口、零 worker DataLoader、梯度累积尾批、CPU BF16 autocast、验证损失加权、optimizer/scheduler checkpoint 恢复、旧权重文件读取、流式数据尾部同步逻辑、音频 TAR 和 Emilia 文本过滤。
- 训练、单卡推理、并行推理与 run.sh 的 `--help` 可用，源码编译和 shell 语法检查通过；twinlakes 活跃源码已无 DeepSpeed/ZeRO 或外部消息通知引用。
- 中英文各32条真实配对打包为4个 TAR（10,373,120 字节）。128段音频均保持源文件字节，64对全部能解码，并经过真实 Qwen tokenizer、filter、padding 得到16个 batch。见 [试打包验证](outputs/shard_trial_20261027/verification.json)。没有启动全量打包。
- 先前受限执行环境禁止 socket 创建；用户调整权限后已完成实际 GPU codec 编码验证，真实多进程 DDP/GPU 训练仍未验收。DDP 训练在任一 rank 数据耗尽时共同结束该轮，舍弃其他 rank 的剩余数据，以保持 optimizer/scheduler 更新次数一致。

当前仍需处理的缺口（本次清理未修改模型算法）：

- 当前 `conf/run_stage1.yaml` 使用 VibeVoice head，用户已将训练 forward 更新为原生 `noisy_images/timesteps/condition` 调用。本次修复了索引赋值 dtype 不匹配，并将有效 speech frames 展开为 `[N,D]` 与条件逐帧对齐；真实小配置 Qwen3 + VibeVoice head 在 FP32/BF16/FP16、原始音频/latent 六种组合下均完成前向和反向。大模型/GPU 训练仍需用户环境验证；采样函数仍有旧 LLM head 逻辑，尚未完成生成验收。
- 当前训练使用真实LLM hidden condition，目标仍是DDPM v-prediction，不是本计划的linear FM；一阶声学MLP负分支已接入forward，两路loss按0.8/0.2加权。CFG推理、流式可见范围和采样路径还需按计划统一，此次未更换DDPM目标。
- 当前训练输入完整目标文本，尚未恢复原计划的流式可见范围，也未验证可懂度、零样本克隆、EOS 或音质。
- 原有 [19项骨架验证](outputs/twinlakes_validation/report.json) 和旧 FM 原型验证只作为历史结果。对应过时测试已备份，并替换为当前 API 的测试，不将历史通过数算入当前验证。

配套 codec 的 [6条重建验证](outputs/codec_validation/report.json) 仍作为历史兼容性证据：原配重建平均 SIM=0.9266，直接混用 Realtime decoder 为0.0515，不能将重建指标当作新模型的克隆指标。

数据 SIM 全量统计见 [报告](outputs/emilia_sim_extremes/report.json)：英文18,116,577条、中文19,636,245条中，精确等于1或大于1均为0；最高分别为0.999372、0.999578。SIM≥0.999共有39/36条，它们不是字节或解码 PCM 完全相同的文件，但部分等长波形余弦极高，仍需排查近重复或重叠；本次未据此改动全量训练清单。

## 6. 当前代码与运行

### Emilia acoustic latent 离线提取

使用当前 TwinLakes 配套的 `/nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B`，仅加载 acoustic encoder，不加载 LLM 或 decoder。与 raw 训练相同，音频先经 torchaudio 解码、声道平均、重采样到24 kHz和 `AudioNormalizer(-25 dBFS)`；完整音频逐条编码，不截断、不混用其他音频的 padding。保存未缩放的 encoder mean，不进行 posterior sampling。默认模型、输入和输出均为 BF16。

```text
原始：.../Emilia/EN/EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.mp3
本地英文：/data/joe/data/latent/EN/EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.latent
本地中文：/data/joe/data/latent/ZH/ZH_B00000/ZH_B00000_S00000/mp3/ZH_B00000_S00000_W000000.latent
完成后归档：本仓库 data/Emilia/latent/EN/...、data/Emilia/latent/ZH/...
```

按用户最新方案，默认输出根目录改为各机器的本地 `/data/joe/data/latent`，可用 `--output` 覆盖；先按大写 `EN/`、`ZH/` 分语言，再保留原相对目录。`.latent` 内容为 `torch.save` 保存的 CPU Tensor `[T,64]`，无 batch 维或 padding；`T = ceil(24k采样数 / 3200)`。使用 `torch.load(path, map_location='cpu', weights_only=True)` 读取。训练中仍应且只应执行一次 `(latent + bias) * scale`。根目录 `_codec.json` 记录 encoder 权重 SHA256、实现 SHA256、音频预处理、dtype 和缩放协议，续跑遇到不一致将拒绝混写。修改输出布局后，已经运行的旧进程需要重启才能使用新的路径。

```bash
# 本机已经验证的解释器；可通过此变量覆盖 shell 脚本中的默认路径。
export EXTRACT_PYTHON=/data/joe/anaconda3/envs/vibe/bin/python
# 默认 GPU 4；先在独立目录提取英文、中文各2条，保留完整音频。
bash tools/extract_emilia_latent.sh --limit 2 --output /data/joe/data/latent_smoke --log-interval 1
# 验证后全量提取原始 en/zh，输出到本机 /data/joe/data/latent。
bash tools/extract_emilia_latent.sh
# 如需指定多卡，一张卡一个独立进程，不建立 DDP 训练进程组：
CUDA_VISIBLE_DEVICES=4,5 bash tools/extract_emilia_latent.sh
# 每卡2个编码进程，8卡共16个 rank，所有 rank 共同按行分片，不重复提取：
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROCESSES_PER_GPU=2 bash tools/extract_emilia_latent.sh --workers 4
# 重叠 GPU 编码与 NFS 写入；只运行此命令中的一组全量任务：
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROCESSES_PER_GPU=4 bash tools/extract_emilia_latent.sh --workers 4 --write-workers 2
```

每个进程流式扫描 manifest，以行序号取模分工，内存只保留有界音频预取；不会建立三千多万条的内存列表。`PROCESSES_PER_GPU` 默认为1，可经 shell 入口设置2、4或8；`--workers` 是每个编码进程的音频解码/预取线程数。`--write-workers` 默认为0（同步），设为2时每个进程使用2个后台线程写入，队列最多16条，仅保留CPU latent。只在原子写入成功后计入 written；队列有界，退出前等待写完，任何写入异常都会传播并阻止标记 complete。

rank 统计新增 `input_wait_seconds`、`encode_and_copy_seconds`、`write_seconds`、`write_wait_seconds`：分别累计等待音频/索引（含缓存查找）、GPU编码及拷回CPU/校验、写入线程的累计墙钟耗时、主进程等待写入的耗时。异步情况下写线程与编码重叠，write_seconds 可以超过总运行时间，不能将各项比例直接相加；encode_and_copy 包含传输和 GPU 等待，不能当作纯 kernel 时间。

文件先写同目录 `.partial` 再原子改名，续跑默认跳过非空成品；`--verify-existing` 逐个加载校验已有文件，异常文件重新提取。解码失败记入 `_runs/<run_id>/rank-xxx.errors.jsonl` 并继续其余音频，该次执行最终非零退出；GPU/写盘错误直接退出。各 rank 的统计和完成状态保存在对应 JSON，只有所有 rank 均 complete 且未设 `--limit` 才代表本次两种语言全部完成。单文件缓存尚未接入 `Dataset(raw)` 的自动读取，不会仅因生成文件就自动改变训练输入。

### 当前全量分工：172英文、27中文

按用户最新安排，两台机器分别运行独立任务：172只读取 `en.jsonl`，写入本机 `/data/joe/data/latent/EN/`；27只读取 `zh.jsonl`，写入本机 `/data/joe/data/latent/ZH/`。shell 入口支持首个位置参数 `en` 或 `zh`，等价于指定 `--languages`；不读取另一种语言的 manifest。两台机器各自在本机的64个 rank 内分片，共用相同 codec 协议，但本地输出互相独立，运行统计记录 hostname、languages。已有本地成品会跳过；切换目录后不会自动把CFS上的已有文件当作本地进度。用户已停止两台原有全量任务，本会话仅进行有界测试，不代为重启全量。

```bash
# 172机器，vibe环境：只提英文。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROCESSES_PER_GPU=8 \
EXTRACT_PYTHON=/data/joe/anaconda3/envs/vibe/bin/python \
  bash tools/extract_emilia_latent.sh en --workers 4 --write-workers 2

# 27机器，vibe环境：只提中文。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROCESSES_PER_GPU=8 \
EXTRACT_PYTHON=/data/joe/anaconda3/envs/vibe/bin/python \
  bash tools/extract_emilia_latent.sh zh --workers 4 --write-workers 2
```

两台机器的 vibe 解释器与8张CUDA设备均已检查可用；语言隔离和shell参数转发已通过测试。按用户要求，本会话修改代码与文档，不代为启动全量。

### 本地提取完成后同步到CFS

两台机器的 `/data` 均为本地ext4（`/dev/md0p1`）。检查时172剩余约15TB、可用inode约3.82亿；27剩余约8.9TB、可用inode约3.74亿，足够对应语言缓存。原CFS目录实际为两台共享的Lustre卷；同步运行时观察到修改RPC槽位等待，每个MDT客户端 `max_mod_rpcs_in_flight=8`。本地写入可以避免该步骤的共享元数据竞争，输入MP3仍从CFS读取。

提取完成后在各自机器手动执行下列同步入口，校验 codec 元数据一致后复制对应语言，保留本地缓存、跳过CFS已存在文件和未完成的 `.partial` 文件：

```bash
# 172机器：英文归档到本仓库 data/Emilia/latent/EN/
bash tools/sync_emilia_latent.sh en
# 27机器：中文归档到本仓库 data/Emilia/latent/ZH/
bash tools/sync_emilia_latent.sh zh
```

源路径可通过 `EXTRACT_LOCAL_DIR` 覆盖，归档根目录可通过 `EXTRACT_CFS_DIR` 覆盖。回传大量小文件仍受CFS元数据性能影响，不将本地提取吞吐当作端到端完成时间；如果后续采用shard训练，可以先在本地打包后回传，减少共享盘文件数量。

两台同时运行的有界对照（每台对应语言8,192条，每卡8进程、解码4线程、后台写2线程，音频预热；两台模型均就绪后同步起跑）：

| 主机与数据 | CFS写入条/秒 | 本地写入条/秒 | CFS平均GPU利用率 | 本地平均GPU利用率 |
| --- | --- | --- | --- | --- |
| 172英文 | 197.7 | 363.2 | 48.9% | 93.8% |
| 27中文 | 287.5 | 388.2 | 71.7% | 95.7% |

两轮保持原逐行分片，均关闭目录缓存以单独比较存储位置。新增的目录准备LRU缓存默认开启；`--partition-mode block --partition-block-size 1000` 为可选分块分工，可减少多个rank同时写同一说话人目录，但尚未做性能验收，不将其算成本次提速来源。原有全量任务保持停止，对照结束后测试进程全部退出。

记录：[172 CFS](outputs/emilia_dual_node_20261027/baseline_172/report.json)、[27 CFS](outputs/emilia_dual_node_20261027/baseline_27/report.json)、[172本地](outputs/emilia_dual_node_20261027/local_172.report.json)、[27本地](outputs/emilia_dual_node_20261027/local_27.report.json)。每台8,192个文件均完整落盘，分别抽查覆盖64个rank的64条，本地与CFS基准逐元素一致。9项提取协议测试通过；同步脚本的小样本检查验证字节保持、本地保留、跳过已有目标、排除partial和拒绝不一致codec，见 [同步检查](outputs/emilia_dual_node_20261027/sync_verification.json)。

### 27机器的提取性能测试

2026-10-27 在 `joe@10.126.203.27`、`/data/joe/anaconda3/envs/vibe/bin/python`（PyTorch 2.4.1+cu121）、A800 80GB 上实测。只测试有限样本，输出均在 `outputs/emilia_latent_benchmark_27_*`，不启动全量。吞吐排除模型加载，计入编码、读音频和写 NFS；输入 MP3 预热到文件缓存。GPU 利用率为 nvidia-smi 定期采样的均值，下面不同样本量的数值不能当作严格的速度比。

| GPU数 | 每卡编码进程 | 每进程解码线程 | 每进程异步写线程 | 中英合计样本 | 平均GPU利用率 | 整机条/秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 4 | 0 | 512 | 53.3% | 33.4 |
| 1 | 1 | 16 | 0 | 512 | 50.7% | 32.7 |
| 1 | 2 | 4 | 0 | 512 | 78.8% | 44.8 |
| 1 | 4 | 4 | 0 | 512 | 94.9% | 51.3 |
| 8 | 4 | 4 | 0 | 4,096 | 69.3% | 298.8 |
| 8 | 4 | 4 | 2 | 4,096 | 80.7% | 362.0 |
| 8 | 8 | 4 | 2 | 8,192 | 94.2% | 380.8 |

单卡增加解码线程没有收益。8卡32进程同步写入时，写文件占各 rank 累计墙钟耗时约44%；改为后台写入后，同一批数据吞吐约提高21%。每卡8进程、2个后台写线程的8卡测试达到约94%平均GPU利用率，单卡峰值显存约19,620 MiB，整机64个编码进程。若目标是较低的CPU/显存占用，可以使用每卡4进程；若要接近打满本次测试的8卡，使用下列配置。整套命令启动一次，不要重复启动几套竞争同一输出。

```bash
export EXTRACT_PYTHON=/data/joe/anaconda3/envs/vibe/bin/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROCESSES_PER_GPU=8 \
  bash tools/extract_emilia_latent.sh --workers 4 --write-workers 2
```

记录：[单卡对照](outputs/emilia_latent_benchmark_27_20261027/report.json)、[8卡同步写](outputs/emilia_latent_benchmark_27_8gpu_20261027/report.json)、[8卡异步写](outputs/emilia_latent_benchmark_27_8gpu_async_20261027/report.json)、[8卡每卡8进程](outputs/emilia_latent_benchmark_27_8gpu_p8_20261027/report.json)。单卡各配置的512个实际输出文件均完整，抽查中英16条逐元素相同；异步写入的4,096条均落盘，抽查64条与同步写结果逐元素相同。多进程映射、分片、异步排空与异常传播等6项测试通过；这不是长程全量提取的吞吐承诺。

VS Code 调试使用 [.vscode/launch.json](.vscode/launch.json) 的 `Python: Debug`：显式固定解释器 `/data/joe/anaconda3/envs/vibe/bin/python`，通过 `module: twinlakes.bin.train` 直接调试单卡训练。`program` 指向某个环境内的脚本并不会选择该环境的 Python。已确认默认 base Python 缺少 diffusers，而 vibe 内的 diffusers 0.35.2 可导入本地 VibeVoice DPM scheduler；依赖已记入 requirements。此检查只验证环境与导入，没有启动真实模型训练或 VS Code GUI 调试会话。

2026-10-28依赖清单更新：[requirements.txt](requirements.txt)按172已运行训练的vibe环境固定直接依赖版本（Python 3.11.14、torch/torchaudio 2.4.1、transformers 4.57.3、diffusers 0.34.0），补齐tensorboardX、tensorboard、tqdm及VibeVoice音频文件接口使用的librosa。前述本机Python 3.12/torch 2.7.1/diffusers 0.35.2属于另一套验证环境，不再作为该文件的版本基准。未改动任一现有环境；该清单不是包含全部传递依赖的pip freeze锁文件。

新机器建议使用Python 3.11；先安装匹配的CUDA torch/torchaudio，再安装其余依赖：

```bash
python -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

GNU sort、NVIDIA驱动及所选音频解码后端属于系统依赖，不由requirements安装。全局随机打包需要GNU sort；原始音频/MP3流程需要可用音频后端（如FFmpeg），离线latent训练不解码音频。APEX、flash-attn、DeepSpeed不是当前训练必需项。安装依赖不等同于完成当前尚待对齐的CFG推理功能。

172验证：13个直接依赖固定版本均与现有vibe环境一致，训练入口/LMModel/Dataset/scheduler导入及TensorBoard事件写入通过，torch实际build为2.4.1+cu121。见 [验证记录](outputs/dependency_audit_20261028/verification.json)。此次未执行全新环境安装验证，也未修改当前训练环境。

```text
run.sh / conf/run_stage1.yaml
  → twinlakes.bin.train
  → LMModel.from_audio_text_pretrained
  → Dataset → processor → padding → DataLoader
  → Executor.train/cv
  → Adam + WarmupLR + 原生 .pt/.yaml checkpoint
```

保留 `bin/train.py`、`bin/infer.py` 和调用同一推理实现的 `bin/infer_parallel.py`；模型为 `models/rq_transformer.py`。数据处理保留 `dataset.py`、`processor.py`、`datapipes.py`、`audio_shard.py`；工具仅保留 `executor.py`、`checkpoint.py`、`scheduler.py`、`common.py`、`train_utils.py`。

按用户最新要求，配置 `dtype` 直接决定模型浮点参数、浮点 buffer 和模型浮点输入的类型：`fp32/bf16/fp16` 对应 `torch.float32/bfloat16/float16`。训练和推理的 `--dtype` 可显式覆盖配置；模型 factory 和训练入口都应用该 dtype，训练入口不再调用 `model.float()`。Executor 在 train/cv 中递归移动数据并转换浮点张量，`input_ids`、labels、lengths、audio_pos、bool mask 保留原始整数或布尔类型。保留对应 dtype 的 autocast，但不使用 FP32 master parameters；直接 FP16 参数不兼容原生 GradScaler 的 unscale，因此 scaler 在此模式下禁用。loss 的显式 FP32 计算保留。只将可训练参数交给 optimizer，codec 冻结。step 表示 optimizer 更新次数，尾部不足 accum_grad 的 microbatch 仍正确平均后更新。配置可覆盖 dtype、accum_grad、grad_clip、max_steps、save_interval、log_interval。 最新 dtype 验证记录见 [configured_dtype 报告](outputs/configured_dtype_20261027/report.json)：19项测试通过，包含 checkpoint 恢复、三种 dtype 的浮点输入转换、整数类型保留，以及真实小配置模型的六种前向/反向组合；这不代表已完成 GPU 大模型训练或 FP16 长程稳定性验收。

完成模型与采样协议对齐并建立独立验证集后，单卡/多卡使用同一入口，例如：

```bash
TTS_PYTHON=/data/joe/anaconda3/envs/vibe/bin/python
TTS_PYTHON="$TTS_PYTHON" train_data=data/Emilia/train.jsonl \
  cv_data=/path/to/heldout.jsonl model_dir=outputs/native bash run.sh

TTS_PYTHON="$TTS_PYTHON" nproc_per_node=2 \
  train_data=/path/to/train.list data_type=shard \
  cv_data=/path/to/heldout.list model_dir=outputs/native_ddp bash run.sh
```

`raw` 读取配对 JSONL 并在线编码；支持 Emilia 相对目标路径和参考 ID。`shard` 在 `dataset_conf.shard_latent: false` 时读取音频 TAR；为 true 时读取下述自包含二进制 latent TAR，同时兼容旧 `.text/base64(npy)` latent TAR。配置 `latent_codec_metadata` 时校验新格式的 codec 来源、dtype 和未缩放协议，旧格式因缺少来源信息会拒绝混入；训练入口同时核对缓存维度及模型 scale/bias。源级 shuffle 使用有界缓冲，避免一次载入全量 JSONL。

### 在172本地打包 latent 并直接训练

用户已将中文特征拷到172，打包使用该机器本地 `/data/joe/data/latent`。旧版按原顺序打包已完成27,806,940对、2,781个TAR；本次按用户要求改为全局随机后重新分包。输入使用已过滤配对的 `data/Emilia/train.jsonl`，不使用缺少prompt的原始 `en.jsonl`/`zh.jsonl`。目标 wav 相对路径和 prompt ID 均映射到 `latent/EN/`、`latent/ZH/` 中相应 `.latent`。

`tools/make_shard.py` 现在默认 `--format latent`。每个样本在TAR中按顺序存3个成员：

```text
<id>.json     原文本、prompt文本、配对信息、codec签名和帧数
<id>.latent   原始目标 .latent 的序列化字节
<id>.prompt   原始prompt .latent 的序列化字节
```

保留BF16 `[T,64]` 原始encoder mean，不转base64、不转FP32、不重新编码或缩放。每个配对自包含，训练不再打开分散的小latent文件。完整 `_codec.json` 另存于shard目录；每个样本带紧凑codec签名，读取时检查shape/dtype/有限值/未缩放协议。训练仅在模型内部应用一次 `(latent+bias)*scale`，不运行codec encoder。默认每10,000对一个TAR；可调worker数和每包条数，最多排队2倍worker数的块，不加载全量JSONL。latent打包建议 `--executor process` 使用CPU多进程，`--num_threads` 参数名保留，表示并发worker数；默认thread模式兼容已有音频打包。

2026-10-28新增打包前全局随机：默认 `--shuffle --seed 20261028`，为每条完整配对记录生成128位随机键，经GNU sort在本地磁盘排序后再切包。不会拆散target/prompt、丢样本或重采样；中英文和说话人按原始数据分布混合，不强制每包固定比例。相同输入与seed可复现顺序，换seed产生新顺序；`metadata.json`记录随机算法、seed及总数。`--no-shuffle`可保留原始顺序。`--max-samples N`取全局随机后的前N条，因此仍先扫描排序全量输入。

随机排序使用 `--shuffle-memory 1G` 作为sort缓冲预算，临时文件放 `--shuffle-temp-dir /data/joe/data/Emilia_latent_shards`，不在CFS上随机读取原JSONL。当前约16.4GiB输入建议预留至少60GiB本地临时空间；正常完成或Python异常退出会清理本次临时目录。开始时先全量排序，日志每100万条报告进度，排序结束后才开始写TAR。此次尚未执行全量重打包。

本次验证：13项打包测试通过，覆盖全局混合、相同seed复现、不同seed换序、样本/配对完整性、关闭随机及排序失败清理。172的vibe环境使用真实EN/ZH各256对生成16个TAR，每包均包含两种语言；512对无丢失或重复，target/prompt及文本正确，特征payload与原文件字节一致；同seed的线程与多进程结果逐包SHA256一致。见 [测试日志](outputs/shuffle_packing_20261028/tests.log) 和 [真实数据验收](outputs/shuffle_packing_20261028/verification.json)。

在172的vibe环境执行正式打包（此命令本会话未启动）：

```bash
cd /nfs-speech-cfs/wangzhou/tts/VibeVoice_Realtime
/data/joe/anaconda3/envs/vibe/bin/python tools/make_shard.py \
  --format latent --latent-root /data/joe/data/latent \
  --executor process --num_utts_per_shard 10000 --num_threads 64 \
  --shuffle --seed 20261028 --shuffle-memory 1G \
  --shuffle-temp-dir /data/joe/data/Emilia_latent_shards \
  data/Emilia/train.jsonl \
  /data/joe/data/Emilia_latent_shards/train_shuffled \
  /data/joe/data/Emilia_latent_shards/train_shuffled.list
```

打包会检查目标与参考特征、文本、dtype和codec协议；缺失或错误立即报错，报告 `metadata.json` 保持complete=false，不发布可用于训练的成功list。TAR先写 `.partial` 再原子发布，拒绝覆盖已有输出；失败修复后使用新的输出目录。正式打包完成后，可直接使用 [latent配置](conf/run_stage1_latent.yaml)：

`run.sh` 本次设置为stage=2、stop_stage=2，只执行64进程打包；在172的vibe环境运行 `bash run.sh 1 0`。新输出使用 `train_shuffled/` 和 `train_shuffled.list`，保留旧 `train/`；训练阶段的list路径及 `conf/run_stage1_latent.yaml` 的codec元信息路径也已指向新数据，打包完成后将stage和stop_stage改为3再训练。本次未停止或重启现有训练任务。

历史性能对照（未开启全局随机）：2026-10-28在172检查旧任务发现64线程仅使用约1.4个CPU核；真实16,384对样本对照（每包256对）中，64线程总耗时19.6秒，16进程总耗时5.3秒，去掉启动后的打包吞吐约917→11,160对/秒。两种方式的64个TAR逐包SHA256完全相同。小样本热缓存结果不代表全量冷读及随机打包吞吐。见 [性能对照](outputs/latent_shard_trial_20261028/packing_benchmark.json)、[字节一致性](outputs/latent_shard_trial_20261028/packing_process_correctness.json) 和 [含多进程CLI的测试](outputs/latent_shard_trial_20261028/process_pool_tests.log)。

```bash
CUDA_VISIBLE_DEVICES=0 /data/joe/anaconda3/envs/vibe/bin/python -m twinlakes.bin.train \
  --config conf/run_stage1_latent.yaml --data_type shard \
  --train_data /data/joe/data/Emilia_latent_shards/train_shuffled.list \
  --model_dir /data/joe/data/exp/vibevoice_latent \
  --num_workers 4 --pin_memory
```

该配置已指定 `shard_latent: true`、BF16和shard目录的 `latent_codec_metadata`。移动TAR后需要更新list中的绝对路径及配置里的codec元数据路径。二进制latent训练读取采用严格模式，丢失、损坏或协议不一致会报错，不静默丢样本。processor已将30秒上限修正为225帧（24k/3200），并允许机器缺少可选libsox时导入，离线latent不依赖SoX。

按用户要求，processor已移除声纹embedding读取、相似度计算/字段、阈值参数及batch声纹补零；Dataset、Executor、latent解码器和LMModel同步移除这些字段与未使用的传参。训练条件继续使用参考音频/latent上下文。历史TAR中的额外声纹字段会被忽略；离线数据准备中既有的SIM配对与过滤记录仍保留。当前33项测试通过，包含原始音频和latent的三种dtype前向/反向，见 [清理测试](outputs/remove_speaker_features_20261028/tests.log)。

2026-10-28 在172实际验收：中英文各16对，4个TAR共1,013,760字节；每个payload与源 `.latent` 字节完全相同，双worker DataLoader读取32条、16个batch，无重复/缺失，BF16、帧数和EOS/目标label mask正确。实际Qwen3-1.7B+当前VibeVoice head完成一次BF16前向/反向及SGD诊断更新，loss=4.1324，encoder调用被禁止仍通过，连接器输入证实只缩放一次，冻结codec无梯度。此验收未运行全量打包或正式Adam训练，也不将数据接通视作FM/流式算法已完成。见 [打包记录](outputs/latent_shard_trial_20261028/packing.json)、[训练链路验收](outputs/latent_shard_trial_20261028/verification.json) 和 [测试日志](outputs/latent_shard_trial_20261028/unit_tests.log)。

训练启动时，`train.py` 在冻结参数并应用配置 dtype 后，由 rank 0 打印配置文件、LLM/codec 初始化路径、实际 LLM/head 维度和完整模型结构；参数表列出各顶层模块及 codec encoder/decoder 的总参数、可训练参数、冻结参数和 dtype。`TOTAL` 按共享 Parameter 去重，codec 子项为拆分统计，不与父项重复相加。新增统计与 rank 0 日志测试已通过；本次训练入口测试共11项，其中10项通过，既有投影测试因当前模型将 `diffusion_head_proj` 设为 `None` 而失败，详见 [验证日志](outputs/model_summary_20261028/tests.log)。本次未启动或重启正式训练。

Checkpoint 保存模型、optimizer、scheduler、scaler 与 epoch/step，另有 YAML 元信息；兼容旧权重-only 文件。没有保存 DataLoader 游标或每 rank RNG，不能保证逐样本/逐位恢复；中途 checkpoint 会重启当前 epoch。旧权重文件恢复时 optimizer 会重新初始化，日志会说明。没有 `--train_engine` 或 DeepSpeed 参数，也没有原 stage 0 离线提取入口。

推理按 JSONL 读取 `id/key`、`text`、`prompt_wav`（路径或 Emilia ID），显式指定 CFG：

```bash
python -m twinlakes.bin.infer --config /path/to/completed.yaml \
  --checkpoint /path/to/model.pt --test_data /path/to/eval.jsonl \
  --result_dir outputs/eval --gpu 0 --dtype bf16 --cfg_scale 2
python -m unittest discover -s tests -v
```

### 音频试打包

[make_shard.py](tools/make_shard.py) 保留三个位置参数与 num_utts_per_shard/num_threads 选项。输入流式读取，按配对而非说话人聚合；有界排队，不载入全量元数据。写入失败会传播异常，不发布成功的 shards list。

每对样本在同一 TAR 中有三个连续成员，共享同一 ID：

```text
<id>.json    文本和原元数据，附 audio_members/audio_formats
<id>.mp3     目标原始音频字节（扩展名随原格式）
<id>.prompt  参考原始音频字节（格式由 JSON 标注）
```

`text`、`prompt_text` 仍是文本；音频不做 base64、不重编码、不重采样。`.prompt` 使用单独扩展名，以兼容 TwinLakes 按最后一个点分组的 TAR DataPipe。此 TAR 存的是音频，不能当成 latent 缓存；训练时仍需 codec 编码。

```bash
python tools/make_shard.py --format audio --num_utts_per_shard 16 --num_threads 2 \
  outputs/shard_trial_20261027/input.jsonl \
  /path/to/new_shards /path/to/new_data.list
```

实际试验清单为 [data.list](outputs/shard_trial_20261027/data.list)，完整字节校验、解码与数据流水线结果见 [verification.json](outputs/shard_trial_20261027/verification.json)。按用户“先试验”的要求，没有全量打包或真实模型训练。

## 7. 维护与实验记录

每次训练必须保存模型与训练配置、代码版本或快照、数据 manifest、随机种子、codec/归一化协议、prompt 配对规则、CFG 与采样设置、checkpoint 路径及评估结果。新结果应同步到本文件的当前状态或明确链接的实验记录中。

调整学习率、batch size、FM 时间采样、dropout、CFG scale、采样步数等属于计划内实验。更换 codec、主生成架构、核心目标或零样本评估协议属于路线变更，应依据用户指示同步修订计划，不能静默替换。

## 参考资料

- [Emilia 数据与处理流程](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia)
- [F5-TTS linear FM 与条件丢弃实现](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [FlowTTS-GRPO](https://arxiv.org/abs/2606.23190)
