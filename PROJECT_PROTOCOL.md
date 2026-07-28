# hanzi_stroke_temporal_composition 科学与实施协议

## 1. 项目边界与权威顺序

本分支从 `codex/digit-writing-original-protocol2` 的正式提交
`f76067232fac8add756b04ea75a8825cca156330` 创建，但项目名称、配置、运行目录、
checkpoint、日志和 artifacts 均使用独立命名空间 `hanzi_stroke_temporal_composition`。
数字项目代码保留用于复用和回归检查，不得把数字 checkpoint、optimizer state、组合系数
或训练结果加载到本项目。

权威顺序如下：

1. `hanzi_writing/hanzi_geometry_final.py`：标准笔画几何、三个汉字布局、物理尺度、速度、弧长采样、真实移笔条件和完整汉字时序的唯一来源；
2. 本文件：模型、输入、损失、训练、checkpoint、冻结验证和服务器运行合同；
3. `CODEX_MINIMAL_FINAL_HANZI_STROKE_COMPOSITION.md`：本轮改造依据；
4. 原仓库实际代码：决定复用部分的真实技术行为；若与论文不同，以原仓库为准。

## 2. 任务与输入

保留 10 个 rule 输入列，使用其中 9 个：

```text
0 heng       1 shu        2 pie       3 na
4 dian       5 ti         6 hengzhe   7 shugou
8 move       9 始终为 0
```

方向条件完全取消。训练和 checkpoint 验证都不旋转轨迹，也不采样方向。

28 维 observation 顺序不变：

```text
[0:10]   rule one-hot
[10:11]  speed scalar
[11:12]  go cue
[12:14]  spatial cue
[14:16]  fingertip visual feedback
[16:28]  muscle proprioceptive feedback
```

书写笔画的 spatial cue 固定为 `[0, 0]`。move 的 spatial cue 是下一笔绝对起点的
character-local 坐标除以权威脚本给出的 `cue_scale_m`；字符名和条件元数据不得进入网络。

## 3. 几何、速度与时序

所有几何只从权威脚本导出。三个完整汉字 `mu`、`jiang`、`ke` 只用于冻结验证，
不得进入训练 sampler。

```yaml
dt_s: 0.01
target_long_medium_steps: 85
train_speed_mps:
  fast: 0.5
  medium: 0.25
  slow: 0.16666666666666667
train_speed_scalar:
  fast: 0.6666666666666666
  medium: 0.3333333333333333
  slow: 0.0
stable_steps: 25
delay_steps: [25, 50, 75]
hold_steps: 25
prepare_steps_in_complete_characters: 25
```

`target_long_medium_steps = 85` 是 2026-07-28 服务器工作空间审计后的用户批准值：初始
150-step 尺度有 92/305 个条件越过 MotorNet 关节限位；逐整数扫描中 1–90 全部通过、
91 首次失败。用户选择 85 而非贴近限位的数学最大值 90；85 的最小关节余量为
0.037491513 rad。未经用户另行批准不得改变该值。

每条笔画和 move 按 `ceil(length / (speed * dt))` 确定 interval 数，再做线性弧长重采样。
movement 同时监督首尾点，共有 `intervals + 1` 个样本。

go cue 的固定语义是：stable/prepare/delay 为 0，movement 为 1，hold 为 0。

## 4. 训练条件与固定随机性

训练只包含 15 个孤立笔画 primitive 和通用 move。每次 update 先在 9 个 rule 中均匀采样，
再在该 rule 内均匀采样条件、三档速度和 delay，禁止直接对全部条件做总体均匀采样。

笔画起点覆盖中心和该 rule 在三个汉字中的全部唯一真实起点。每个 primitive 与该 rule 的
所有起点交叉；每个基础起点增加 4 个逐轴独立的均匀 jitter：

```yaml
fraction_of_global_character_span: 0.03
distribution: uniform_per_axis
copies_per_base_start: 4
seed: 42
```

move 覆盖 12 个真实 transition 和权威脚本按 seed 42 生成的 4 份 jitter，共 60 个条件。
所有已解析策略和种子必须写入运行配置、condition manifest 和日志，不允许静默默认值。

## 5. 模型、损失与优化

继续使用项目 2 的 CPU 基准：

```yaml
device: cpu
network: rnn
input_size: 28
hidden_size: 256
activation: softplus
output_size: 6
recurrent_noise_std: 0.1
input_noise_std: 0.01
constrained: false
rnn_dt_ms: 10
rnn_tau_ms: 20
batch_first: true
optimizer: Adam
learning_rate: 0.001
batch_size: 32
max_updates: 75000
validation_interval: 500
grad_clip_norm: 1.0
l1_rate: 0.001
l1_weight: 0.001
l1_muscle_act: 0.01
simple_dynamics_weight: 0.001
```

位置误差 `e_t = |x_t-x*_t| + |y_t-y*_t|`。四阶段先分别取 mean，再计算：

```text
L_position = 0.1 L_stable + 0.1 L_delay + 0.6 L_movement + 0.2 L_hold
L_total = L_position + L_rate + L_weight + L_muscle + L_simple_dynamics
```

基础训练使用完整五项损失；checkpoint 选择只使用 `L_position`。

## 6. Checkpoint 验证

每 500 updates 使用固定网格：15 个 primitive 与其中心/全部真实 rule 起点、12 个 exact move、
三档速度、delay 50、无 jitter、无完整汉字。共 81 个 rollout group。

验证 seed 为 1042，保留原仓库的网络噪声行为。先在每个 rule 内平均，再对 9 个 rule 等权平均；
条件较多的 heng、shu、move 不得获得更高权重。

## 7. 完整汉字冻结验证

训练完成后，分别以 medium 速度运行 `mu`、`jiang`、`ke`。每个汉字是一个连续 trial：
隐藏状态和手臂动力学状态都不在笔画之间重置，rule 随时间切换，move cue 随目标变化。

冻结验证必须满足：

- 网络与输出层参数不求梯度，不创建或 step optimizer；
- 验证前后 policy `state_dict` bitwise 相同；
- 不存在可训练的外部组合系数；
- writing mask 只覆盖书写 movement，不覆盖 move；
- 报告 writing-only mean L1、逐笔 mean/endpoint、move mean/endpoint、final hold drift；
- 输出三个字的 target、pen-up 和 writing 轨迹图。

## 8. 正式训练前服务器审计

新实例视为全新服务器，只在 `/root/autodl-tmp` 下建立独立 CPU 环境、仓库和输出目录。
环境依赖通过国内镜像安装，且必须验证 `torch==2.6.0+cpu`、`cuda_available=False`。

审计必须覆盖：

1. 权威几何/时序 self-test 和三字物理轨迹；
2. 28-D observation、动态 rule、go/cue/mask；
3. 9-rule sampler、stroke jitter 和 81 组 checkpoint 网格；
4. 三个字、全部 isolated stroke placements、全部 exact/jitter move 的 MotorNet workspace；
5. 指定的 heng/shu/na/hengzhe/shugou/move 短闭环；
6. `mu`、`jiang`、`ke` 未训练 deterministic frozen smoke，实际步数必须精确等于权威 schedule 长度；
7. 无 NaN/Inf、episode 正常结束、五项损失可记录、冻结和数据泄漏检查。

工作空间失败时只报告并停止，不得自动缩放、改速度或改模型。

## 9. 配置与入口

```text
configurations/hanzi_stroke_temporal_composition_geometry.json
configurations/hanzi_stroke_temporal_composition_train_dev42.json
configurations/hanzi_stroke_temporal_composition_validate_characters.json

server/create_hanzi_cpu_environment.sh
server/run_hanzi_stroke_temporal_composition_audit.sh
server/run_hanzi_stroke_temporal_composition_train_dev42.sh
server/run_hanzi_stroke_temporal_composition_validate_characters.sh
```

审计脚本可以创建环境并执行所有测试，但不能启动正式训练。正式 75,000-update 训练和训练后的
冻结整字验证均由独立授权标签保护。

当前状态：85-step 尺度已获用户批准，必须重新通过完整服务器审计；尚未启动 75,000 updates 正式训练。
