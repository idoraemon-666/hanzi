# hanzi_stroke_temporal_composition 接续说明

## 当前身份

- 分支：`codex/hanzi-stroke-temporal-composition`
- 来源：`codex/digit-writing-original-protocol2` 的 `f76067232fac8add756b04ea75a8825cca156330`
- 固定 submodule：`mRNNTorch@ac0c4f589eae37bbde63968912925de99232e306`
- 设备协议：CPU；新 AutoDL 卡不参与计算
- 新环境：`/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu`
- 新仓库建议路径：`/root/autodl-tmp/hanzi_stroke_temporal_composition_repo`

## 接手顺序

完整阅读 `AGENTS.md`、本文件、`PROJECT_PROTOCOL.md`、
`CODEX_MINIMAL_FINAL_HANZI_STROKE_COMPOSITION.md` 和
`hanzi_writing/hanzi_geometry_final.py`。只有处理服务器步骤时阅读
`AUTODL_SERVER_USAGE_GUIDE.md`，只采用其中的操作规范，不继承旧项目路径或状态。

## 固定决定

- 复用项目 2 的 28-D observation、RNN 256 softplus、MotorNet、CPU、五项损失和 75k 协议；
- 9 个 active rule，rule 列 9 始终为 0；方向条件取消；
- hold 的 go cue 为 0；
- stroke 起点使用中心、全部真实 rule 起点和 seed 42 的 3% 逐轴均匀 jitter；
- sampler 先均匀选 rule，再在 rule 内选条件/速度/delay；
- checkpoint 固定 15 primitive、12 exact move、三速度、delay 50，共 81 rollout groups；
- checkpoint 有网络噪声，seed 1042，按 9 rule 等权；
- 完整 `mu/jiang/ke` 只做训练后 deterministic frozen validation。

## 下一步

只在服务器执行环境创建和审计。返回审计 `.tar.gz`、`.sha256` 和完整后台日志后再判断是否
允许正式训练。任何 workspace、闭环、NaN/Inf、冻结或测试失败都只报告，不自动改几何、
速度、损失、网络或阈值。

当前状态：实现尚未获得服务器审计证据，且尚未启动 75,000 updates 正式训练。
