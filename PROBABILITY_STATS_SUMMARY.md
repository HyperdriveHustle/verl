# 概率统计实现总结

## 目标与范围

- 记录训练阶段与推理阶段的 token 概率，追加写入到同一 JSONL 文件；不影响训练和推理主流程，仅作为解耦的增量统计。
- 训练概率：取训练时某位置的 `logits`，使用稳定 `softmax` 计算出该位置的具体 token 概率。
- 推理概率：取推理引擎返回的 `logprobs`，对其 `exp` 得到概率，按响应序列位置对齐。

## 关键组件与位置

- 训练端提取与写入：`code/verl/verl/workers/actor/dp_actor.py`
  - `_forward_micro_batch`：前向计算后在安全分支调用统计；移除 padding 路径用 `response_logits`；非移除 padding 路径用 `logits`。
  - `_extract_and_save_training_probabilities`：对每个响应位置，根据 `responses` 的 token id，从该位置 `logits` 计算概率并写入。

- 推理端提取与传递：`code/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
  - `generate`：从 vLLM 的 `RequestOutput` 中解析每步 `logprobs`；兼容字典键为 int/string、对象属性 `logprob` 等格式。

- 异步管道贯通：`code/verl/verl/experimental/agent_loop/agent_loop.py`
  - `_postprocess`：将 `response_logprobs` 拼接为 `rollout_log_probs` 并加入 batch；训练端可用此作为 `inference_probs` 的回退来源（经 `exp`）。

- 通用工具与落盘：`code/verl/verl/utils/probability_extraction.py`
  - `stable_softmax`：稳定的 softmax；训练概率统一使用该函数计算。
  - `extract_token_probability`：从某位置 `logits` 计算给定 token 的概率并取值。
  - `ProbabilityLogger`：以追加模式写入 JSONL；字段包括 `Training Possibility`、`Inference Possibility`、`Token Index`。

## 运行时行为与分支

- 非移除 padding、非 fused kernels：
  - 在 `_forward_micro_batch` 的非 rmpad 分支中，得到 `logits` 后调用统计；推理概率优先使用 `inference_probs`，否则从 `rollout_log_probs.exp()` 回退。

- 使用移除 padding（rmpad）时：
  - 先将 `logits_rmpad` 通过 `pad_input` 复原为 `full_logits`，再切片得到 `response_logits`，随后进行统计；推理概率同样支持从 `rollout_log_probs.exp()` 回退。

- 使用 fused kernels：
  - 不进行训练概率统计（该路径无显式的 `logits`）；避免影响主流程。

## 配置与启用

- 通过配置项开启统计：

```yaml
actor_rollout_ref:
  actor:
    probability_output_file: "/path/to/output/probabilities.jsonl"
  rollout:
    probability_output_file: "/path/to/output/probabilities.jsonl"
```

- 仅在 rank 0 进程写入，避免重复。

## JSONL 字段约定

- `Training Possibility`：稳定 softmax 计算的训练概率，保留 8 位小数。
- `Inference Possibility`：推理 `logprobs.exp()` 的概率，保留 8 位小数；若推理概率不可得则记为 `0.0`。
- `Token Index`：响应序列中的相对位置索引。

## 兼容性与健壮性

- vLLM `logprobs` 解析：
  - 支持字典键为 `int` 或 `string` 的情况；当返回对象含 `logprob` 属性时也能取到数值。
  - 缺失或异常时以 `NaN` 占位并在后续阶段跳过统计或回退。

- 维度与对齐：
  - 所有概率按 `response_length` 切片；`response_mask` 控制有效位置。
  - 训练与推理对齐使用同一响应索引，保证同一位置的 token 比较。

## 不影响主流程的保证

- 统计逻辑在训练损失与反向传播之外执行，不改变梯度与优化器行为。
- fused 路径跳过统计，移除 padding 路径使用复原后的 `response_logits`，避免未定义变量。

## 后续验证建议

- 运行训练，检查输出 JSONL 中 `Inference Possibility` 不为 0 且与训练概率在同一位置有合理一致性。
- 若存在异常值或全为 0，检查 `rollout_log_probs` 是否非空且维度与 `responses` 对齐；同时确认配置已启用 `probability_output_file`。

