# 汉字5：`hanzi_stroke_temporal_composition` 当前正式接管

更新时间：2026-08-03（Asia/Shanghai）

唯一工作仓库：`D:\hanzi_stroke_temporal_composition`

当前代码基线：`e41238ef8dea6bc5d3ba7781ba9beb118086be62`

## 1. 强制作用域

本任务只接管 `hanzi_stroke_temporal_composition`。

- 不得读取、修改、切换、清理、运行或接管
  `D:\digit_writing_original_protocol2`。
- 不得处理 `digit_writing_original_protocol2` 或
  `digit_writing_original_protocol3` 的代码、分支、任务文档、服务器结果或科学问题。
- Codex 默认 cwd 可能显示为无关的数字 worktree；所有 Git、文件、搜索和运行命令
  必须显式指定 `D:\hanzi_stroke_temporal_composition`。
- `protocol3` remote 与本项目完全无关，不查询、不 fetch、不 push。
- 不因共享 Git 仓库中存在数字分支而切换或审查它们。

## 2. 新任务接管顺序

主任务亲自完整阅读，不能把阅读和判断委托给子任务：

1. 本文件；
2. `AGENTS.md`；
3. `DUAL_FIXED_RULE_RNN_PROTOCOL.md`；
4. `DUAL_FIXED_RULE_JOINT_GRADIENT_PROTOCOL.md`；
5. 两个正式配置：
   - `configurations/hanzi_stroke_temporal_composition_dual_fixed_rule_rnn_v1.json`
   - `configurations/hanzi_stroke_temporal_composition_dual_fixed_rule_joint_gradient_v1.json`
6. 当前核心源码：
   - `hanzi_writing/dual_rule_protocol.py`
   - `hanzi_writing/dual_rule_envs.py`
   - `hanzi_writing/dual_rule_training.py`
   - `losses.py`
7. 两个正式服务器入口：
   - `server/run_hanzi_dual_fixed_rule_loss_comparison.sh`
   - `server/run_hanzi_dual_fixed_rule_joint_gradient.sh`
8. 当前回归测试：
   - `tests/test_hanzi_dual_fixed_rule_rnn.py`
   - `tests/test_hanzi_dual_fixed_rule_joint_gradient.py`
   - `tests/test_phase_normalized_loss.py`
9. 几何权威及 canonical 历史：
   - `hanzi_writing/hanzi_geometry_final.py`
   - `hanzi_writing/geometry.py`
   - `hanzi_writing/canonical_protocol.py`
   - `hanzi_writing/canonical_overfit.py`
   - `hanzi_writing/canonical_start_loss_comparison.py`
   - 所有 `canonical` 汉字配置
10. `AUTODL_SERVER_USAGE_GUIDE.md`；
11. `HANZI4_CURRENT_TAKEOVER_98f7e7e.md`，只作历史记录，并使用本文件
    第 11 节覆盖其中的未来计划。

早期 `PROJECT_PROTOCOL.md` 和
`CODEX_MINIMAL_FINAL_HANZI_STROKE_COMPOSITION.md` 解释项目来源和旧 28-D/9-rule
设计。当前 15-rule Stroke RNN、12-rule Move RNN、33/30-D observation、固定时序和
两种 optimizer-step 方案，以当前双 RNN 协议、配置、源码和本文件为准。

## 3. 本地 Git、worktree、remote 与身份

接管文档创建前核验：

```text
path       D:\hanzi_stroke_temporal_composition
branch     codex/hanzi-stroke-temporal-composition
HEAD       e41238ef8dea6bc5d3ba7781ba9beb118086be62
upstream   origin/codex/hanzi-stroke-temporal-composition
origin     https://github.com/idoraemon-666/hanzi.git
status     clean；本交接文档提交后 HEAD 应为 e41238e 的直接后继
submodule  ac0c4f589eae37bbde63968912925de99232e306，detached 且 clean
user.name  idoraemon-666
user.email 3362948112@qq.com
```

最近汉字提交：

```text
e41238e Add joint-gradient dual RNN comparison
7730279 Add dual fixed-rule stroke and move RNN experiment
1ba1382 Add parallel canonical start-loss comparison
de9f4b0 Add Shugou checkpoint selection experiment
e75304c Add Hanzi 4 takeover handoff
98f7e7e Fix checkpoint review test-count gate
2fffb90 Add threshold-free canonical checkpoint review
c01afd9 Add parallel canonical low-LR training lines
```

`7730279` 是 `e41238e` 的祖先，能够 fast-forward。数字分支和挂载在默认 cwd 的数字
worktree 与本项目无关；不要切换、比较、清理或更新。

接管时先执行只读核验：

```powershell
git -C D:\hanzi_stroke_temporal_composition status --short --branch
git -C D:\hanzi_stroke_temporal_composition rev-parse HEAD
git -C D:\hanzi_stroke_temporal_composition rev-parse origin/codex/hanzi-stroke-temporal-composition
git -C D:\hanzi_stroke_temporal_composition submodule status
git -C D:\hanzi_stroke_temporal_composition\mRNNTorch status --short --branch
```

不得为“回到代码基线”丢弃本交接提交；应确认当前 HEAD 是 `e41238e` 的直接后继。

## 4. 编程和协作规范

- 先梳理完整逻辑、明确科学变量和完成标准，再写代码。
- 最少代码、精准改动；不重构无关路径，不增加未要求的抽象和功能。
- 使用 `apply_patch` 修改文件；不清理用户已有改动或历史运行证据。
- 新实验使用独立 variant、配置、输出目录、checkpoint 身份和授权标签。
- 旧正式结果只读并拒绝覆盖；失败现场保留，重试使用新 run 名。
- 变更后运行与风险相称的测试、真实最小 smoke 和 `git diff --check`。
- 未经用户明确要求，不自行设计或启动下一项训练。
- 用户偏好一次性批量交互；服务器命令应合并为短、可审计的一段。
- 不在 AutoDL 交互终端启用裸 `set -euo pipefail`；严格模式只放在脚本内。
- Codex 不直接连接服务器。用户负责上传 bundle/归档和执行 AutoDL 命令；Codex
  负责本地实现、Git、bundle、命令设计和下载结果审查。
- 训练期间不得更新正在运行的服务器仓库。并行新代码必须使用独立仓库，当前
  joint-gradient 实验已经按此规则实施。

## 5. 当前科学问题和固定任务

当前不是旧的“一个 8-rule shared RNN”方案，而是两个互相独立的多任务控制器：

```text
Stroke RNN: 15 个固定 occurrence one-hot，input size 33
Move RNN:   12 个固定 transition one-hot，input size 30
```

每个 one-hot 唯一绑定一个起点、终点、方向、轨迹和 movement duration。实验测试固定规则
多任务拟合，不测试未见轨迹、长度、起点、速度、方向或 delay 泛化。

共同合同：

```text
256-unit softplus RNN，6 muscle outputs，MotorNet closed loop
seed 42，validation seed 1042
dt 0.01 s，stable 25，delay 50，hold 25
speed scalar 0.0（兼容 cue，不代表不运动）
batch/microbatch per rule 1
training/validation network and observation noise disabled
```

Stroke observation：

```text
[0:15] rule one-hot; [15:16] speed; [16:17] go cue
[17:19] spatial cue [0,0]; [19:21] vision; [21:33] proprioception
```

Move observation：

```text
[0:12] rule one-hot; [12:13] speed; [13:14] go cue
[14:16] absolute target cue; [16:18] vision; [18:30] proprioception
```

Stroke 首末点来自权威汉字轨迹；Move 起点是前一笔终点，终点是后一笔起点。MotorNet
reset 到任务精确起点。`hengzhe`、`shugou` 只平滑内部转折，端点不变。

15 个 Stroke movement intervals：

```text
long_heng_ke_0 150; long_heng_jiang_5 126; medium_heng_mu_0 106
medium_heng_jiang_3 75; short_heng_ke_3 45; long_shu_mu_1 150
medium_shu_jiang_4 56; short_shu_ke_1 39; pie_mu_2 150; na_mu_3 150
dian_jiang_0 150; dian_jiang_1 129; ti_jiang_2 150
hengzhe_ke_2 200; shugou_ke_4 200
```

12 个 Move movement intervals：

```text
mu_move_0 125; mu_move_1 161; mu_move_2 147
jiang_move_0 61; jiang_move_1 121; jiang_move_2 56
jiang_move_3 73; jiang_move_4 79
ke_move_0 199; ke_move_1 65; ke_move_2 42; ke_move_3 94
```

## 6. 损失、验证和 checkpoint

每个模型并行比较三种训练位置损失：

```text
baseline      phase-normalized L1
full_trial    full-trial L1
onset_window  加强 delay 末 10 步和 movement 前 5 步
```

正则保持：`l1_rate=0.001`、`l1_weight=0.001`、`l1_muscle_act=0.01`、
`simple_dynamics_weight=0.001`，梯度裁剪 `1.0`。

验证不是独立 held-out 数据：每个 one-hot 只有一条固定目标轨迹。验证关闭噪声、评估所有
规则，只用于 checkpoint 选择和方案比较，不能证明未见轨迹泛化。

三种损失统一用 `macro_mean_phase_normalized_position_l1` 比较。每个 worker 保存：

```text
best_macro_checkpoint.pt
best_worst_checkpoint.pt
final_checkpoint.pt
continuation_checkpoint.pt
```

没有行为通过阈值；compound transition、exit direction 等为描述性指标，不能只凭平均
loss 宣称成功。

## 7. 形成当前方案的已完成证据

八笔画单任务、两阶段 refinement、scratch `1e-4`、额外 continuation 和无阈值
checkpoint review 已完成。结论：两阶段 `1e-3 -> 1e-4` 整体更稳定；final 经常劣于
训练中 best；`shugou` 的总体位置、转折和出口方向存在权衡；单任务 checkpoint 只能作
容量和拟合基线，不能合并成共享 RNN。

Shugou 专项正式提交 `de9f4b0`，归档：

```text
D:\hanzi-canonical-shugou-checkpoint-experiment-de9f4b0-dev42-run1.tar.gz
```

`subphase_equal` 被无阈值多指标规则推荐，但不 Pareto-dominate control。用户随后明确不接受
“只为一个任务改变共享模型损失函数”作为多任务正式方案。

起点损失三方案正式提交 `1ba1382`，归档和轨迹图：

```text
D:\hanzi-canonical-start-loss-comparison-1ba1382-dev42-run1.tar.gz
D:\hanzi_start_loss_comparison_plots_1ba1382
```

报告整体排名是 `full_trial > onset_window > baseline`，但不同笔画冠军不同。用户查看轨迹后
明确选择 `onset_window` 作为后续偏好。这个人工选择不得被整体排名静默覆盖；当前双 RNN
正式比较仍保留三种损失，由实验判断最终方案。

## 8. 当前两组正式多任务实验

### 8.1 一规则一更新的 round-robin 基线

正式 HEAD：`7730279852fb1b4d548218da21eb3ba561959296`

```text
Stroke 120000 updates = 90000 @ 1e-3 + 30000 @ 1e-4
Move    96000 updates = 72000 @ 1e-3 + 24000 @ 1e-4
每规则 8000 exposures；deterministic fixed round-robin
Stroke/Move × 3 loss arms = 6 workers
每 600 optimizer updates 验证
```

服务器：

```text
repo    /root/autodl-tmp/hanzi_stroke_temporal_composition_repo
HEAD    7730279852fb1b4d548218da21eb3ba561959296（启动时）
output  /root/autodl-tmp/hanzi_stroke_temporal_composition_repo/runs/hanzi_stroke_temporal_composition/dual_fixed_rule_loss_comparison/dev42
archive /root/autodl-tmp/hanzi-dual-fixed-rule-loss-comparison-7730279-dev42-run1.tar.gz
log     /root/autodl-tmp/hanzi-dual-fixed-rule-loss-comparison-7730279-dev42-run1.launcher.log
```

### 8.2 每轮跨规则平均梯度的 companion

正式 HEAD：`e41238ef8dea6bc5d3ba7781ba9beb118086be62`

```text
Stroke 每 step 依次计算 15 条规则，各完整 objective / 15 后累积
Move   每 step 依次计算 12 条规则，各完整 objective / 12 后累积
每轮只 clip 一次平均梯度、Adam step 一次；不 padding
两模型各 8000 optimizer steps = 6000 @ 1e-3 + 2000 @ 1e-4
每规则 8000 exposures；总 rule rollouts 与 round-robin 一致
Stroke/Move × 3 loss arms = 6 workers
Stroke 每 40 joint steps、Move 每 50 joint steps 验证
```

服务器独立仓库：

```text
repo    /root/autodl-tmp/hanzi_stroke_temporal_composition_joint_gradient_repo
HEAD    e41238ef8dea6bc5d3ba7781ba9beb118086be62（启动时）
output  /root/autodl-tmp/hanzi_stroke_temporal_composition_joint_gradient_repo/runs/hanzi_stroke_temporal_composition/dual_fixed_rule_joint_gradient_loss_comparison/dev42
archive /root/autodl-tmp/hanzi-dual-fixed-rule-joint-gradient-e41238e-dev42-run1.tar.gz
log     /root/autodl-tmp/hanzi-dual-fixed-rule-joint-gradient-e41238e-dev42-run1.launcher.log
```

## 9. 服务器环境、容量和最后证据

共同环境：

```text
/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
torch 2.6.0+cpu; MotorNet 0.2.0; CUDA false
submodule ac0c4f589eae37bbde63968912925de99232e306
```

容量证据：解除错误外层 OMP 限制后 `nproc=128`；16-worker probe 成功，
`parallel16_sec=45`、相对 single slowdown `1.115x`、总 peak RSS `12.51 GiB`。因此用户
批准同时运行 12 个单线程 worker。

最后有证据的状态（2026-08-01）：

- round-robin 六个 worker 正式启动；
- joint-gradient 通过 `102` 项汉字测试和 `4` 项损失测试；
- joint-gradient 六个 worker 全部启动；
- 启动命令确认原 6 + 新 6 = 12 个训练 worker；
- 未启动完整汉字训练或 Stroke/Move 串联控制器。

截至 2026-08-03，本地 D 盘没有上述两组正式结果 `.tar.gz`/`.sha256`，只有上传 bundle。
因此只能证明“成功启动”，不能证明“仍在运行”或“已经完成”。需要时给用户一段只读、短、
不会关闭终端的批量检查命令，核验进程、launcher 完成/失败标记、worker summary/runtime、
归档和 SHA。不要直接连接服务器，也不要重启、覆盖或清理运行目录。

## 10. Bundle 与结果审查

```text
D:\hanzi-dual-fixed-rule-loss-comparison-7730279.bundle
SHA256 6aac44eff912f6cc98be1a051f94efc30a9d79969a43dd85be859ac68414beee

D:\hanzi-dual-fixed-rule-joint-gradient-e41238e.bundle
SHA256 1c475f628eeb4c14a88419c27bda74000022a7f6c05779f946b6a434e78627a8
```

服务器 GitHub fetch 可能卡住，优先 bundle。同步前检查 branch/HEAD/submodule/clean；能
fast-forward 时只用 `merge --ff-only`。并行实验必须独立 clone，不更新活动训练仓库。

两组新归档下载后必须：外部 SHA 匹配、独立解包、内部 `SHA256SUMS` 全通过、退出码为 0、
skipped 为 0、provenance 完整。还要比较服务器产生的 condition manifest，确认两方案使用
相同规则顺序、轨迹和时序；不要用 Windows 本地浮点 manifest hash 替代服务器同平台比较。
然后再比较三种损失、macro/worst/final checkpoints 和逐规则轨迹图。

## 11. 已过期或被覆盖的事项

`HANZI4_CURRENT_TAKEOVER_98f7e7e.md` 中以下未来计划已过期：

- “move 暂不处理”已改为独立 12-rule Move RNN；
- “8-rule shared RNN、64k global updates”未实施，已被 15-rule Stroke RNN 与
  12-rule Move RNN 取代；
- 多 delay `[25,50,75]` 已改为单一 `delay_steps=50`；
- 旧 28-D observation 已被 Stroke 33-D、Move 30-D observation 取代；
- 用户曾撤销一版八任务设计，不能恢复那版脚本；
- round-robin batch-1 基线之外，已正式启动跨全部规则平均梯度 companion。

不冲突且继续有效：单任务 checkpoint 只是拟合基线，不能拼接成共享模型；checkpoint
选择必须同时看平均、最差规则和轨迹行为指标。

## 12. “汉字5”首次动作与报告

新任务创建后只做只读接管，不修改代码、不启动实验、不连接服务器。完整阅读本文件和
第 2 节材料后，向用户简洁报告：

1. 已锁定唯一汉字仓库并忽略两个数字任务；
2. branch/HEAD/upstream/submodule/clean；
3. 15-rule Stroke、12-rule Move、三种损失和固定时序；
4. round-robin 与 joint-gradient 的准确差别和公平口径；
5. 两个服务器 repo、共同 env、输出、归档和 bundle 方式；
6. 最后证据只是 12 workers 成功启动，完成状态尚未核验；
7. 单任务、Shugou 和起点损失比较的关键结论；
8. 是否发现事实冲突。

无冲突则等待用户下一条指令。不要自行恢复旧 shared-8 设计或启动新训练。
