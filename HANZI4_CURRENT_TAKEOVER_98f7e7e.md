# 汉字4：`hanzi_stroke_temporal_composition` 当前正式接管

更新时间：2026-07-31
唯一工作仓库：`D:\hanzi_stroke_temporal_composition`

## 1. 强制作用域

本任务只处理 `hanzi_stroke_temporal_composition`。

- 不得修改、切换、清理或接管 `D:\digit_writing_original_protocol2`。
- 不得处理 `digit_writing_original_protocol2` 或
  `digit_writing_original_protocol3` 的需求、分支、服务器结果或任务文档。
- `D:\digit_writing_original_protocol2` 当前实际挂载的是无关的 protocol-3
  worktree；不要因为 Codex 默认 cwd 指向它而执行任何命令。
- 所有 Git、搜索、读取和写入命令必须显式作用于
  `D:\hanzi_stroke_temporal_composition`。
- `protocol3` remote 与本任务无关，不查询、不 fetch、不 push。

## 2. 新任务接管顺序

主任务亲自完整阅读并核验：

1. 本文件；
2. `AGENTS.md`；
3. `PROJECT_PROTOCOL.md`；
4. `CODEX_MINIMAL_FINAL_HANZI_STROKE_COMPOSITION.md`；
5. `D:\CODEX_CANONICAL_STROKE_OVERFIT_GATE.md`；
6. `hanzi_writing/hanzi_geometry_final.py`；
7. `hanzi_writing/geometry.py`、`envs.py`、`training.py`、`audit.py`、
   `fit_diagnostic.py`；
8. `hanzi_writing/canonical_protocol.py`、
   `canonical_overfit.py`、`canonical_refinement.py`、
   `canonical_parallel_lines.py`、`canonical_checkpoint_review.py`；
9. `configurations/` 下全部 canonical 汉字配置；
10. `AUTODL_SERVER_USAGE_GUIDE.md`；
11. 本文件第 7 节登记的正式归档及最新比较报告。

`PROJECT_PROTOCOL.md` 包含早期一般训练协议。当前 canonical 单轨迹实验的已批准
语义以 `D:\CODEX_CANONICAL_STROKE_OVERFIT_GATE.md`、当前 canonical 配置、当前源码
和本文件第 5、8 节为准。发现冲突时报告，不静默折中。

## 3. 本地 Git 身份与状态

Git identity：

```text
user.name  = idoraemon-666
user.email = 3362948112@qq.com
```

当前汉字 worktree（`98f7e7e` 是新增本交接文档前的代码基线；本交接文档提交后，
实际 HEAD 应是它的直接后继，接管时必须以只读 Git 命令重新核验）：

```text
path       D:\hanzi_stroke_temporal_composition
branch     codex/hanzi-stroke-temporal-composition
code HEAD  98f7e7e6f02fe5b6c5f9cb4aee6a13e1c91011c0
upstream   origin/codex/hanzi-stroke-temporal-composition
status     本交接文档提交并推送后应 clean
submodule  ac0c4f589eae37bbde63968912925de99232e306
```

相关本地分支身份（只读登记，不得接管数字分支）：

```text
codex/hanzi-stroke-temporal-composition  98f7e7e 的直接后继  当前唯一工作分支
codex/digit-writing-original-protocol2   f760672  无关
codex/digit-writing-original-protocol3   552b20f  无关，挂载在默认 cwd
local/digit-writing-original-protocol2-folder 15b46c7 无关
```

remote：

```text
origin     https://github.com/idoraemon-666/hanzi.git
protocol3  https://github.com/idoraemon-666/digit1.git  # 完全忽略
```

最近汉字提交：

```text
98f7e7e Fix checkpoint review test-count gate
2fffb90 Add threshold-free canonical checkpoint review
c01afd9 Add parallel canonical low-LR training lines
d763b59 Add canonical eight-stroke refinement gate
6b2007b Run canonical single-task gate in parallel
2cf251c Split canonical Hanzi Stage 0 approval gate
147d3e6 Add canonical Hanzi single-task overfit gate
07b6e01 Add fixed-duration 10k pilot execution and diagnostics
```

## 4. 服务器协作方式与最后有证据的状态

不要由 Codex 直接连接服务器。用户在 AutoDL 终端执行一次性、可复制的批量命令；
Codex 负责本地代码、Git/bundle、命令设计和下载归档审查。

服务器：

```text
repo   /root/autodl-tmp/hanzi_stroke_temporal_composition_repo
env    /root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu
branch codex/hanzi-stroke-temporal-composition
last evidenced HEAD 2fffb90e394b2fa91c44d7d88b3457385358c68b
submodule ac0c4f589eae37bbde63968912925de99232e306
```

服务器仓库来自旧卡的完整复制，硬件型号相同。`origin` 已配置，但服务器访问 GitHub
会长时间卡在 `git fetch`。不要继续等待或反复 fetch；优先生成小型增量 Git bundle，
让用户通过 AutoDL 文件界面上传到 `/root/autodl-tmp`，再执行本地 bundle fetch 和
`merge --ff-only`。

当前本地已准备、尚需在下一次服务器工作前同步的修复 bundle：

```text
D:\hanzi-checkpoint-review-fix-98f7e7e.bundle
D:\hanzi-checkpoint-review-fix-98f7e7e.bundle.sha256
```

它只包含 `2fffb90 -> 98f7e7e`，修复正式比较 wrapper 的测试计数
`80 -> 108`。科学比较结果由 `2fffb90` 产生且有效；服务器代码尚停在 `2fffb90`。

服务器命令必须：

- 不使用会在失败时关闭用户终端的外层裸 `set -e`；
- 用子 shell 捕获退出码，末尾明确打印 `TERMINAL_REMAINS_OPEN=1`；
- 运行前检查 branch/HEAD/submodule/clean status；
- 目标输出只要已存在就拒绝覆盖；
- 正式运行使用独立授权标签；
- 输出 `execution.log`、`exit_code.txt`、`provenance.json`、`SHA256SUMS`、
  `.tar.gz` 和外部 `.sha256`；
- 不删除、移动或覆盖既有正式 runs。

用户偏好一次性批量交互，不要逐条让用户复制命令或反复回报状态。

## 5. 当前 canonical 科学语义

当前已取消：

- 多长度；
- 多起点和 jitter；
- 多速度；
- 多方向；
- stroke target-position cue；
- `move`（当前下一阶段明确排除）。

八个 stroke：

```text
heng shu pie na dian ti hengzhe shugou
```

每个 rule 只有一个最长 canonical occurrence、一个 exact 原始起点、一个方向和一个
固定 duration。用户已经人工批准 `hengzhe` 和 `shugou` 的理论目标轨迹。

固定输入/时序：

```text
simple strokes (heng/shu/pie/na/dian/ti): 150 movement intervals, 151 samples
compound strokes (hengzhe/shugou):        200 movement intervals, 201 samples
stable_steps: 25
delay_steps: [25, 50, 75]
hold_steps: 25
speed_name: slow
speed scalar input: 0.0
stroke spatial/target-position cue input: [0, 0]
```

重要澄清：

- speed scalar 的 `0.0` 是 slow 档的兼容 cue，不代表实际轨迹不运动；
- 实际 target trajectory 非零，保存在环境中用于监督 loss，不作为二维目标位置输入；
- 网络仍输入 rule one-hot、go cue、当前 fingertip visual feedback 和 proprioception；
- 单任务训练的 delay 是 seed 42 下每 update
  `random.choice([25, 50, 75])`，不是严格均衡循环；
- checkpoint validation 固定 delay 50；
- refinement/follow-up 精确保留并恢复 optimizer、global RNG 和 environment RNG。

当前 `HanziComponentEnv` 明确要求同一 batch 共享 rule、speed 和 movement duration，
且整批 rule one-hot 由第一条 trajectory 填充。因此：

- 不得直接构造“六个简单笔画 batch 6 + 两个复合笔画 batch 2”；
- 直接这样做会在 `_require_batch_compatible` 失败；
- 粗暴取消检查会把整批错误标成第一个 rule，科学结果无效；
- mixed-rule batch 需要新的隔离实现和验证，用户当前没有批准。

## 6. 已完成实验与科学结论

### 6.1 Stage 0 与 Stage 1

- canonical Stage 0 目标审计通过；
- 9 个单任务最初包含 8 stroke + move；
- 9 任务最终并行运行到每任务 6000 updates、LR `1e-3`；
- 用户后来明确 `move` 暂不处理。

### 6.2 八笔画 refinement

从 Stage 1 的正式候选继续：

```text
8 strokes
exact optimizer/RNG continuation
learning rate 1e-4
additional 2000 updates/task
8 processes
```

两阶段 `1e-3 -> 1e-4` 对 `shu`、`dian`、`hengzhe` 明显优于从头低学习率，
整体也更可靠。

### 6.3 两条并行低学习率线路

1. 八笔画从头训练：LR `1e-4`，每任务 10000 updates；
2. `heng/pie/na/shugou` 从 refinement final 精确续跑 2000，
   refinement 累计达到 4000。

从头 `1e-4` 并不整体优于两阶段训练。8 个 scratch 最终 review 全部劣于各自训练中
best，说明 final state 不稳定，checkpoint 选择必要。

续跑结论：

- `heng`：有收益，但 final endpoint 变差；
- `pie`：best 有收益，final 回退；
- `na`：没有明确 validation 收益；
- `shugou`：总体位置 loss 改善，但 transition 与 exit direction 逐步恶化。

### 6.4 无阈值 checkpoint 比较

正式比较：

```text
variant     canonical_checkpoint_metric_review_v1
git HEAD    2fffb90e394b2fa91c44d7d88b3457385358c68b
tests       108 passed, 0 skipped
sources     4
candidates  56
leaders     58
behavioral_pass_fail_defined              false
automatic_checkpoint_selection_performed  false
manual_review_required                     true
```

外部 SHA256：

```text
0101043b2fba60748d08c70ab8edfe8cd9fa8402e09877d702fa08bf2c6e3ec3
```

核心证明：最低 position validation loss 不保证 endpoint、path ratio、compound
transition 或 exit direction 同时最佳。`shugou` 最清楚：

```text
follow-up best u3500:
  validation 0.004429
  transition 3.011 mm
  exit direction 2.555 deg

follow-up review u3999:
  validation 0.004001  # 更低
  transition 5.357 mm  # 明显更差
  exit direction 7.279 deg  # 明显更差
```

八个独立 checkpoint 不能合并成一个共享 RNN；它们只用于单任务容量证据和共享模型
基线，不是 shared 模型初始化组合。

## 7. 正式本地证据

所有归档及 `.sha256` 位于 `D:\`：

```text
hanzi-canonical-single-task-overfit-6b2007b-dev42-parallel6000-run1.tar.gz
SHA256 51138e0c88faa119445d0cd8d5d86d698726a3186f5bc8f06b1b92d70907d311

hanzi-canonical-eight-stroke-refinement-d763b59-dev42-lr1e4-2000-run1.tar.gz
SHA256 a527c61a081e11bce7cb395b096d6504ab6327145bc28f48be11f44ea3020afa

hanzi-canonical-scratch-lr1e4-c01afd9-dev42-10000-run1.tar.gz
SHA256 35f3025a0823cdf09850452ad153dd1d287635ddba21959f959aea2b093469c8

hanzi-canonical-selected-followup-c01afd9-dev42-2000-run1.tar.gz
SHA256 8ba193892b75dcc815daec0eed1e37aee2f55b8654079231acb0e328a0a32e35

hanzi-canonical-checkpoint-metric-review-2fffb90-dev42-run1.tar.gz
SHA256 0101043b2fba60748d08c70ab8edfe8cd9fa8402e09877d702fa08bf2c6e3ec3
```

最新比较的只读解包目录：

```text
D:\hanzi_checkpoint_review_audit_0101043b\dev42
```

其中：

```text
CHECKPOINT_METRIC_REVIEW_REPORT.md
checkpoint_candidate_metrics.csv
checkpoint_metric_leaders.csv
source_manifest.json
provenance.json
execution.log
exit_code.txt
SHA256SUMS
```

外部 SHA 匹配，7 个内部 SHA 全部通过。

## 8. 用户已批准的下一实验

下一项不是继续审查，而是正式设计并运行一个 shared-8 多任务模型：

```text
one shared RNN policy
active rules: 8 strokes only
move: excluded
batch_size: 1
total global updates: 64000
expected task exposure: about 8000 updates/task
fresh seed: 42
phase 1: 48000 global updates at learning rate 1e-3
phase 2: exact continuation for 16000 global updates at learning rate 1e-4
```

训练条件继续严格使用第 5 节 canonical 单轨迹语义。用户选择该方案是为了最大限度
降低欠训练风险，接受预计约 28–36 小时墙钟时间。

任务调度必须保证八任务公平。先前讨论倾向 deterministic rule round-robin；delay
必须保持与单任务一致的 seeded random choice `[25,50,75]`，不得改成均衡循环。
如果当前权威材料仍不能唯一决定 task scheduler，设计脚本前只就这一点向用户确认，
不要自行引入 mixed-rule batch、padding 或梯度累积。

shared-8 的一个 global update 只训练一个 task。batch 32 在当前 canonical 确定性
单条件语义下只是同一任务的重复样本，不能用更少 updates 等价替代；用户已拒绝将
`batch32 + 少量 updates` 作为正式方案。

## 9. 下一任务的执行边界

新任务首先完成只读接管确认，然后直接进入 shared-8 设计，不再新增无必要的审查项目。

必须：

- 最小、隔离地新增显式 shared-8 variant；
- 不修改旧 Stage 0/Stage 1/refinement/scratch/follow-up 结果语义；
- 不修改 RNN、MotorNet、28-D observation、已有 loss 项或 loss 权重；
- 不加入 move、多长度、多起点、多速度、多方向、jitter、padding；
- batch size 保持 1；
- 两阶段使用同一个 shared policy，并精确保留 Adam/RNG/environment RNG；
- 结果目录全新且拒绝覆盖；
- 支持中断后精确恢复；
- formal launch 继续使用独立用户授权标签；
- 只运行与代码安全直接相关的既有/新增测试，不再构造额外科学 preflight 层；
- 给用户一条批量 Git bundle 同步/启动命令，不逐条交互。

在用户再次明确批准服务器正式启动之前，可以完成本地实现、测试合同、提交、push 和
离线 bundle；不得自行连接服务器或启动 64k 正式训练。

## 10. 接管确认格式

新“汉字4”任务完成阅读后，向用户简洁报告：

1. 已锁定唯一仓库和忽略的数字任务；
2. 本地 Git/branch/HEAD/submodule；
3. 服务器 repo/env/最后 HEAD 与 bundle 协作方式；
4. canonical 输入、时序和 batch 限制；
5. 已完成的四轮单任务/精修/并行低 LR/比较证据；
6. 当前核心科学问题；
7. 已批准 shared-8 的 batch1、64k、两阶段方案；
8. 尚未启动 shared-8；
9. 是否发现事实冲突。

无冲突则等待用户在“汉字4”任务中下达 shared-8 脚本设计指令。
