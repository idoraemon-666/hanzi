# hanzi_stroke_temporal_composition

这是从正式项目 2 提交 `f760672` 分出的独立汉字基本笔画时序组合项目。原数字代码保留，
但本项目不读取数字 checkpoint、optimizer state 或训练结果。

核心设计：一个共享 RNN–MotorNet 模型学习 8 种孤立书写笔画和 1 种通用 move；冻结后通过
随时间切换 rule、go cue 和 move goal cue 连续写出 `mu`、`jiang`、`ke`。完整汉字不进入训练。

权威文件：

- `PROJECT_PROTOCOL.md`：科学和实施合同；
- `hanzi_writing/hanzi_geometry_final.py`：几何、尺度、采样、move 和整字 schedule 的唯一来源；
- `CODEX_MINIMAL_FINAL_HANZI_STROKE_COMPOSITION.md`：本轮实施依据。

主要入口：

```text
configurations/hanzi_stroke_temporal_composition_geometry.json
configurations/hanzi_stroke_temporal_composition_train_dev42.json
configurations/hanzi_stroke_temporal_composition_validate_characters.json

server/run_hanzi_stroke_temporal_composition_audit.sh
server/run_hanzi_stroke_temporal_composition_train_dev42.sh
server/run_hanzi_stroke_temporal_composition_validate_characters.sh
```

环境和所有项目测试只在服务器执行。审计会使用独立 CPU 环境，并检查几何、时序、28-D
输入、sampler、81 组 checkpoint 网格、MotorNet 工作空间、短闭环和冻结整字 rollout。

当前停止线：尚未启动 75,000 updates 正式训练；正式训练必须在服务器审计通过并经用户
明确授权后单独启动。
