# Codex 最简实施指令：将数字书写项目改造为汉字基本笔画时序组合项目

> 本文件与随附的 `hanzi\_geometry\_final.py` 共同构成本轮改造的唯一实施依据。
> `hanzi\_geometry\_final.py` 是最终笔画几何、汉字布局、物理尺度、速度、分段采样、移笔条件和完整汉字时序的唯一权威来源。
> 本轮只实现必要且最小的代码改动及相关审查；用户审查前禁止启动正式训练。

\---

## 0\. 项目目标、结论边界与停止线

### 0.1 科学问题

训练一个共享参数的闭环 RNN–MotorNet，使其学习：

```text
横、竖、撇、捺、点、提、横折、竖钩、通用移笔
```

共 9 个 rule。

冻结网络后，通过时间上切换：

```text
笔画 rule
→ 通用移笔 rule + 下一笔绝对起点提示
→ 下一笔 rule
```

生成训练阶段从未完整出现过的三个汉字：

```text
木、江、可
```

本项目验证的是：

> 网络能否复用已经学会的笔画和移笔计算，在外部给定笔顺、切换时刻及下一笔起点的条件下，完成新的时序组合。

不得把结果表述为：

* 网络自主规划或理解了汉字；
* 网络自行推断了下一笔位置；
* 网络实现了真实三维抬笔；
* 网络学习了自然手写速度或书法笔锋。

### 0.2 必须明确的边界

这不是严格意义上的“只换静态轨迹”。多笔画汉字要求以下必要改动：

1. rule input 在完整汉字验证过程中随时间切换；
2. 增加一个通用 `move` rule；
3. 原 2-D spatial cue 在 move 阶段改为下一笔绝对起点；
4. 完整汉字 trial 包含多个笔画、移笔和准备阶段；
5. 增加 writing mask，移笔轨迹不计入字迹。

除此之外，网络、MotorNet、输出、反馈、优化器和基础正则不重新设计。

### 0.3 停止线

完成代码、测试、工作空间审查和短 smoke test 后停止。

用户明确授权前禁止：

* 启动 75,000 updates 正式训练；
* 自动修改几何；
* 自动缩放物理尺度；
* 自动修改三档速度；
* 自动修改网络规模或损失系数；
* 根据 smoke test 结果更换汉字或笔画；
* 启动旧数字项目的代数组合、heldout5 或 transfer 流程。

\---

## 1\. 开始前只做轻量检查

1. 阅读当前 `PROJECT\_PROTOCOL.md`、`CODEX\_MINIMAL\_FINAL\_DIGIT\_MODIFICATION.md` 和当前实际代码；
2. 查看 `git status`、`git diff`；
3. 定位：

   * 数字几何入口；
   * 环境 reset/step；
   * 28-D observation 拼接；
   * rule input；
   * speed scalar；
   * go cue；
   * spatial cue；
   * phase-normalized position loss；
   * 训练采样器；
   * workspace audit；
4. 输出一份计划修改文件清单后直接实施，不等待额外确认。

不要重复：

* Git provenance 全审计；
* 原仓库历史审计；
* 已完成的数字几何来源审计；
* 无关旧图和旧分析；
* 多 seed 正式训练；
* 75,000 updates 训练。

\---

## 2\. 项目隔离

新项目名称固定为：

```text
hanzi\_stroke\_temporal\_composition
```

要求：

* 使用独立配置、运行目录、checkpoint、日志和 artifact 命名空间；
* 不读取或覆盖数字项目训练结果；
* 不加载数字项目 checkpoint、optimizer state 或组合系数；
* 原数字项目代码尽量保留，通过新增汉字环境与入口实现；
* 不删除已完成数字项目文件。

\---

## 3\. 最终几何与数据权威

将随附：

```text
hanzi\_geometry\_final.py
```

整合为推荐路径：

```text
hanzi\_writing/hanzi\_geometry\_final.py
```

可以增加薄适配层：

```text
hanzi\_writing/geometry.py
```

但仓库中不得复制第二套汉字坐标公式。

必须直接复用脚本中的：

* 8 种书写笔画；
* 木、江、可最终布局；
* 横折 80° 向左下直线；
* 竖钩向左上、与竖约 60°；
* 木字最终撇、捺曲线；
* 12 个真实移笔转换；
* 全局物理缩放；
* 三档速度；
* 完整汉字时序生成；
* self-test 和审计输出。

禁止依据文字重新手写坐标。

运行：

```bash
python hanzi\_writing/hanzi\_geometry\_final.py \\
  --self-test \\
  --target-long-medium-steps 150 \\
  --output-dir artifacts/final\_hanzi\_geometry
```

脚本当前自检应得到：

```text
overall\_passed = true
penup\_transition\_count = 12
longest medium intervals = 150
```

\---

## 4\. 9 个 rule 与 28-D 输入合同

### 4.1 rule 映射

保留原有 10 个 rule 输入列，不改变总 observation 维数。

固定：

```text
rule\[0] = 横 heng
rule\[1] = 竖 shu
rule\[2] = 撇 pie
rule\[3] = 捺 na
rule\[4] = 点 dian
rule\[5] = 提 ti
rule\[6] = 横折 hengzhe
rule\[7] = 竖钩 shugou
rule\[8] = 通用移笔 move
rule\[9] = 永远为 0 的保留列
```

不得：

* 为短、中、长横分别新增 rule；
* 为不同相对位移分别新增 move rule；
* 增加汉字 one-hot；
* 增加笔画序号输入；
* 增加字符身份输入；
* 增加轨迹编码器、Transformer 或新网络层。

### 4.2 observation 顺序保持 28 维

```text
\[0:10]   9 个有效 rule one-hot + 1 个保留零列
\[10:11]  speed scalar
\[11:12]  go cue
\[12:14]  2-D spatial cue
\[14:16]  fingertip visual feedback
\[16:28]  muscle proprioceptive feedback
```

### 4.3 spatial cue 新语义

书写笔画阶段：

```text
spatial cue = \[0, 0]
```

移笔准备和移笔运动阶段：

```text
spatial cue = next\_stroke\_absolute\_start\_xy / cue\_scale\_m
```

目标在整个单次移笔阶段保持不变。

`cue\_scale\_m` 必须由权威脚本计算，保证两维均落在 `\[-1, 1]`。

不得只靠 `move` one-hot 猜测位移。

\---

## 5\. 速度、尺度、时间与方向条件

### 5.1 三档名义物理路径速度

保持数字项目协议：

```yaml
dt\_s: 0.01
train\_speed\_mps:
  fast: 0.5
  medium: 0.25
  slow: 0.16666666666666667
train\_speed\_scalar:
  fast: 0.6666666666666666
  medium: 0.3333333333333333
  slow: 0.0
```

这些是模拟器中的名义物理路径速度，不解释为真实人类书写速度。

### 5.2 全局尺度

脚本以：

```text
最长书写笔画在中速条件下 = 150 intervals
```

计算唯一全局尺度。

当前脚本输出：

```text
global\_scale\_m\_per\_design\_unit
= 0.0004385964912280702
```

所有笔画、汉字布局和移笔距离统一乘同一个尺度。

禁止：

* 笔画特异缩放；
* 汉字特异缩放；
* 为通过工作空间审查自动缩小某个字；
* 不经用户批准改变 150-step 目标。

### 5.3 采样

每段按真实弧长独立计算：

```text
intervals = ceil(length\_m / (speed\_mps \* dt\_s))
```

要求：

* 线性弧长重采样；
* 同一段起点和终点均被监督；
* 连接点只保留一次；
* 不使用 minimum-jerk；
* 不添加显式减速；
* 不强制转角速度归零；
* 横折和竖钩内部转折连续；
* 不插入未经规定的轨迹停顿。

\---

5\.4 方向
## 方向条件取消



本项目取消原仓库的方向条件：



\- 不保留 8 个训练方向；

\- 不保留 32 个验证方向；

\- 不对基本笔画或完整汉字做整体旋转；

\- 不进行方向随机采样、方向泛化验证；

\- batch 内不再存在样本间方向差异；

\- 所有笔画和“木、江、可”均按权威几何脚本中的标准正立方向执行；

\- 工作空间审查只检查标准正立方向下的全部笔画、移笔轨迹和三个完整汉字；

\- 三档速度条件继续保留。



原输入 `\[12:14]` 不再表示运动方向：



\- 书写笔画阶段固定为 `\[0, 0]`；

\- 通用移笔阶段表示下一笔绝对起点，经 `cue\_scale\_m` 归一化。



不得因为取消方向条件而删除 `\[12:14]` 两维输入，也不得把横折、竖钩内部的几何夹角误认为原仓库的方向条件。

## 6\. 横、竖长度条件不是新任务

最终字形中的横、竖长度不是只有三个完全相等的数值。为了不再修改用户已确认字形：

* `heng` 始终只有一个 rule；
* `shu` 始终只有一个 rule；
* 训练采样覆盖最终三个字中实际出现的全部横、竖持续时间；
* 短、中、长只用于分组和报告，不生成新的 one-hot。

因此：

```text
同一 heng rule + 同一速度 + 不同持续时间
→ 不同长度横画
```

```text
同一 shu rule + 同一速度 + 不同持续时间
→ 不同长度竖画
```

训练配置中保存所有实际 required duration，禁止将它们拆成多个任务身份。

\---

## 7\. 基础训练任务

### 7.1 单一共享模型

只训练一个模型：

```text
9-rule shared RNN–MotorNet
```

不再建立：

* full10；
* heldout5；
* digit composition；
* digit transfer。

网络结构、MotorNet、输出层、激活函数、反馈通道、噪声、optimizer、learning rate、gradient clipping、基础正则及其系数保持当前项目配置。

### 7.2 书写笔画训练

每个书写 trial 仍是单个基本笔画，而不是完整汉字。

权威脚本提供每个最终笔画实例的：

* relative primitive；
* 真实长度；
* 三档速度时间步；
* 在完整汉字中的实际起点。

为避免完整汉字验证时出现未见过起点的高风险，训练必须覆盖：

```text
中心起点
+
该 rule 在木、江、可中实际使用的所有起点
+
小幅起点 jitter
```

但所有平移版本仍共享同一个 rule 和同一个相对轨迹。

这不会新增任务身份。它只使笔画计算能从不同身体状态和工作空间位置被调用。

训练阶段不得提供汉字身份或完整笔顺。

### 7.3 通用 move 训练

move 是一个参数化任务：

```text
move rule + 下一笔绝对起点 cue
```

训练必须覆盖：

1. 权威脚本给出的 12 个真实笔画间转换；
2. 脚本固定种子生成的轻微起点/目标 jitter；
3. 三档速度。

move 的起点与目标均来自连续坐标，禁止为每个位移建立独立 rule。

最终绘图和 writing metrics 中，move 路径必须被 writing mask 排除。

### 7.4 采样平衡

先均匀采样 9 个 rule，再在 rule 内采样：

* 长度；
* 起点；
* 速度；
* delay；
* move 目标。

禁止直接对全部条件均匀采样，否则横、竖和 move 的条件数量较多，会压制其他笔画任务。

\---

## 8\. 单组件 trial 时序与损失

书写笔画和 move 的基础训练仍使用当前项目的：

```text
stable → delay → movement → hold
```

保持：

```yaml
stable\_steps: 25
delay\_steps: \[25, 50, 75]
hold\_steps: 25
```

训练和 checkpoint 继续使用当前已经实现并通过测试的 phase-normalized position L1：

```text
0.1 stable
+ 0.1 delay
+ 0.6 movement
+ 0.2 hold
```

其他正则项及其完整 trial 语义不变。

不要重新改造损失，除非当前汉字环境无法直接调用既有实现。

\---

## 9\. 完整汉字冻结验证

### 9.1 禁止完整汉字训练

训练 sampler 中不得出现：

```text
木的完整 rule sequence
江的完整 rule sequence
可的完整 rule sequence
```

完整汉字只在网络训练完成并冻结后执行。

### 9.2 每个汉字一个完整 trial

固定三个验证 trial：

```text
木
江
可
```

第一笔起点均由脚本整字平移至工作空间中心。

完整 trial 的时序由 `assemble\_character\_schedule()` 生成：

```text
initial stable
→ 第一笔 prepare
→ 第一笔 movement
→ move prepare
→ move movement
→ 下一笔 prepare
→ 下一笔 movement
→ ...
→ final hold
```

规则：

* 每次 rule 切换后先提供 25-step prepare，go cue = 0；
* movement 开始时 go cue = 1；
* move prepare/movement 持续提供下一笔绝对起点 cue；
* stroke prepare/movement 的 spatial cue 固定为 0；
* move 和准备阶段 writing mask = false；
* 仅笔画 movement writing mask = true；
* 网络隐藏状态在整个汉字 trial 中连续，不重置；
* 物理手臂状态不瞬移；
* 网络、输出层和 MotorNet 全部冻结；
* 不优化新参数或外部组合系数。

### 9.3 主要指标

每个汉字完整报告：

* writing-only mean position L1；
* 每笔 mean position L1；
* 每笔 endpoint error；
* move-only mean position L1；
* 每次移笔 endpoint error；
* final hold drift；
* 笔画顺序正确率；
* rule schedule 与预期完全一致；
* writing mask 下的最终轨迹图；
* 包含虚线 move 路径的诊断图。

不得只展示成功汉字。

\---

## 10\. 已识别的高风险及对应处理

### 风险 A：单个 move rule 无法决定去哪

处理：

```text
move rule 决定“执行移笔”
2-D absolute goal cue 决定“移动到哪里”
```

### 风险 B：基本笔画只在中心训练，完整汉字中起点分布外

处理：

训练覆盖中心、全部实际笔画起点和轻微 jitter；仍共享同一 rule。

### 风险 C：完整汉字中的 rule 切换导致隐藏状态来不及重组

处理：

每次 move 和下一笔之前加入 25-step prepare，保持原仓库 delayed-go 逻辑。

### 风险 D：move 轨迹被误当成笔迹

处理：

显式 writing mask；move 只参与控制误差，不进入字形图和 writing-only 指标。

### 风险 E：任务可能退化为外部控制器直接规划完整汉字

处理与结论限制：

* 外部确实提供 rule 顺序、切换时刻及下一笔起点；
* 网络负责复用与执行笔画/移笔计算；
* 结论限定为“受外部指令调用的时序组合”，不宣称自主规划。

### 风险 F：150-step 物理尺度可能超出 MotorNet 工作空间

这是当前唯一不能只靠几何脚本消除的实施门槛。

必须先做工作空间审查。若任何轨迹不可达、触碰关节限位或 inverse/forward kinematics 不一致：

```text
立即停止并报告
```

禁止自动改尺度。由用户决定是否降低最长中速步数。

### 风险 G：完整 trial 约 900 steps，可能出现长时隐藏状态漂移

处理：

* 冻结 deterministic smoke test；
* 逐段记录误差；
* 每次切换前 prepare；
* 检查 NaN/Inf；
* 不在完整汉字 trial 上反向传播。

\---

## 11\. 只做必要审查

### A. 权威几何与时序

运行：

```bash
python hanzi\_writing/hanzi\_geometry\_final.py \\
  --self-test \\
  --target-long-medium-steps 150 \\
  --output-dir artifacts/final\_hanzi\_geometry
```

必须保存：

```text
hanzi\_geometry\_audit.json
hanzi\_geometry\_final.json
stroke\_manifest.csv
penup\_transitions.csv
character\_schedules\_medium.json
final\_hanzi\_trajectories\_physical.png
```

### B. observation 与 schedule 测试

必须验证：

* observation 始终为 28 维；
* rule 区域始终 10 维；
* rule\[9] 永远为 0；
* 书写阶段 spatial cue 为 0；
* move 阶段 cue 在 `\[-1,1]`；
* move rule 和目标 cue 同时存在；
* time-varying rule schedule 与脚本完全一致；
* writing mask 与 move mask 不重叠；
* 完整汉字所有数组长度一致；
* 每个 movement 首尾目标点均被监督；
* 转换边界没有 off-by-one。

### C. 几何测试

必须验证：

* 三个汉字第一笔起点为 `(0,0)`；
* 横严格水平；
* 竖严格竖直；
* 横折是一笔连续轨迹；
* 横折第二段为向左下直线；
* 横折与横的锐角为 `80° ± 0.2°`；
* 竖钩起点位于顶部长横下方；
* 钩为向左上直线；
* 钩与竖的夹角为 `60° ± 0.2°`；
* 全部坐标有限；
* 最长笔画中速为 150 intervals；
* 共有 12 个真实移笔转换。

### D. 工作空间审查

重新运行与新项目直接相关的审查：

```text
3 characters
+
所有 isolated stroke placements
+
所有 exact/jitter move conditions
```

检查：

* Cartesian bbox；
* radial reach margin；
* joint angle margin；
* inverse/forward kinematics consistency；
* 是否触碰关节限位；
* 第一笔中心起点；
* 最远横、最深竖和最长移笔。

失败时只报告，不得自动缩放。

### E. 短闭环 smoke test

不启动正式训练。只执行极短更新或未训练 rollout 管线检查：

```text
heng: longest, fast/slow
shu: longest, fast/slow
na: medium
hengzhe: medium
shugou: medium
move: 最长 exact transition
move: 一个 jitter transition
完整 木/江/可 deterministic frozen schedule
```

检查：

* 无 NaN/Inf；
* episode 正常结束；
* 约 900-step 完整 schedule 可运行；
* MotorNet 不越界；
* loss 和逐段指标可记录；
* writing mask 正确；
* rule 切换正确。

### F. 冻结与泄漏测试

完整汉字验证前后：

* 网络 state\_dict bitwise 相同；
* optimizer 不存在或不 step；
* 无可训练外部系数；
* 训练 sampler 未生成完整汉字 sequence；
* observation 中不存在 character ID；
* move metadata 中的 character 名称不得进入网络输入。

### G. 不需要重复

不要重做：

* 原始 Git provenance 全审计；
* 数字项目历史安全审计；
* 数字 0–9 全量几何审计；
* 旧 algebraic composition；
* heldout5 transfer；
* 无关旧分析图；
* 正式多 seed 训练。

\---

## 12\. 最小配置与入口

推荐配置：

```text
configurations/hanzi\_stroke\_temporal\_composition\_geometry.json
configurations/hanzi\_stroke\_temporal\_composition\_train\_dev42.json
configurations/hanzi\_stroke\_temporal\_composition\_validate\_characters.json
```

推荐入口：

```text
server/run\_hanzi\_stroke\_temporal\_composition\_audit.sh
server/run\_hanzi\_stroke\_temporal\_composition\_train\_dev42.sh
server/run\_hanzi\_stroke\_temporal\_composition\_validate\_characters.sh
```

配置至少包含：

```yaml
project: hanzi\_stroke\_temporal\_composition
geometry\_source: hanzi\_writing/hanzi\_geometry\_final.py
rule\_count\_active: 9
rule\_dim\_total: 10
unused\_rule\_index: 9
target\_long\_medium\_steps: 150
dt\_s: 0.01
train\_speed\_mps:
  fast: 0.5
  medium: 0.25
  slow: 0.16666666666666667
character\_validation:
  characters: \[mu, jiang, ke]
  speed: medium
  prepare\_steps: 25
  final\_hold\_steps: 25
  freeze\_network: true
  writing\_mask\_enabled: true
```

基础网络和训练超参数继续使用当前项目值，除非现有配置路径必须换成新命名空间。

\---

## 13\. 交付内容

Codex 完成后提交：

1. 修改文件清单；
2. 与本文件逐项对应的差异说明；
3. 权威几何脚本整合位置；
4. 几何和时序 self-test；
5. 28-D observation 与动态 rule schedule 测试；
6. 9-rule 训练 sampler 测试；
7. 12 个真实 move 转换及 jitter 条件表；
8. 3 个汉字物理轨迹图；
9. 工作空间审查；
10. 短闭环 smoke test；
11. 完整汉字冻结测试；
12. 训练、审查、冻结验证入口；
13. 更新后的 `PROJECT\_PROTOCOL.md`；
14. 明确声明：

```text
尚未启动 75,000 updates 正式训练。
```

\---

## 14\. 最终实施原则

优先复用当前数字项目已经通过测试的：

* 28-D observation；
* RNN–MotorNet 闭环；
* 三档物理速度；
* 线性弧长重采样；
* phase-normalized position loss；
* checkpoint 和日志框架；
* workspace audit；
* deterministic freeze test。

只新增完成汉字时序组合不可缺少的部分：

```text
最终笔画几何
9-rule 映射
通用 move task
时变 rule schedule
move goal cue
writing mask
完整汉字冻结 rollout
```

不要重新设计项目。
