# 更新日志（Changelog）

本项目所有值得记录的版本更迭都写在这里。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## 版本约定

当前处于 `0.x` 阶段，约定与主项目 RPGBar 一致：

- **新功能**：`0.MINOR.0`，次版本号 +1。
- **修 bug / 小改动**：`0.MINOR.PATCH`，修订号 +1。
- **破坏性变更**：`0.x` 阶段也要在小节里显式标注 `⚠ 破坏性`，并写明迁移方式。

版本号的**唯一来源**是 `src/engine.cc` 里的 `kEngineVersion`（`rpg_engine_version()`
返回它），与 `CMakeLists.txt` 的 `project(VERSION)` 必须一致。

发版流程（每次都要走完）：

1. 开发时把改动写进下面的 `[未发布]` 小节；
2. 发版时把 `[未发布]` 改成 `## [x.y.z] - YYYY-MM-DD`，并新建一个空的 `[未发布]`；
3. 同步 `kEngineVersion` 与 `CMakeLists.txt` 的版本号；
4. 提交，打 annotated tag：`git tag -a vX.Y.Z -m "..."`；
5. 推送：`git push origin main && git push origin --tags`；
6. 在 GitHub 上对该 tag 创建 Release，正文直接复制本文件里对应版本的小节。

> 记录原则：只写**使用者（宿主开发者）能感知的变化** —— 多了什么、能干什么、
> 行为变了什么。纯内部重构不写。

---

## [未发布]

（暂无）

---

## [0.1.0] - 2026-10-11

首个可用版本。从 GGUF 文件到结构化的工具调用，全链路打通，宿主已经可以真的加载
权重、建会话、生成、释放。

判据一律是"与官方实现对拍"：反量化对 `gguf.quants` 逐位，前向对 PyTorch 官方
`Qwen3ForCausalLM` 逐层，生成对官方逐 token，分词对 HF `tokenizers` 逐 id，
ChatML 渲染对官方 `chat_template` 逐字节。参考侧与引擎用**同一份 GGUF**，
量化误差与实现错误才能分开归因。

### 新增

- **GGUF 解析**：v1/v2/v3 三种版本，元数据与张量清单，张量数据走 mmap 零拷贝。
- **15 种 ggml 量化类型的反量化**：F32/F16/BF16 + `Q4_0`…`Q8_1` + `Q2_K`…`Q8_K`，
  位布局与 llama.cpp 的 `dequantize_row_*` 逐行对齐，实测逐位相同。
- **Qwen BBPE 分词器**：零第三方依赖，手写 qwen2 预分词正则，特殊 token 贪婪匹配；
  3031 条用例与 HF `tokenizers` 逐 id 一致。
- **Qwen3 标量前向**：RMSNorm / QK-norm / RoPE / GQA 因果注意力 / SwiGLU，
  与官方实现逐层对拍（逐元素相对误差 p99 ≤ 2e-4，argmax 全位置一致）。
- **KV Cache 与增量解码**：布局 `[n_layers, n_kv_heads, max_seq, head_dim]`，
  贪心生成与官方实现**逐 token** 一致。
- **ChatML 渲染**：复刻官方 `chat_template` 的分支（含 `last_query_index`、
  连续 tool 消息合并、空思考块），15 个用例逐**字节**与 HF 渲染结果一致。
- **`<tool_call>` / `<think>` 解析**：引擎只解析不执行，输出 `text` /
  `tool_calls` / `finish_reason` / `usage`，被截断时给降级标记而不是抛异常。
- **C ABI（10 个函数）**：两级句柄（`model` 管权重、`session` 管独立 KV Cache 与
  上下文），支持同一权重上多个互不干扰的会话；宿主每轮传完整 messages，引擎自动
  做最长公共前缀匹配复用缓存。

### 已知限制

- **只能贪心**：温度 / top-k / top-p 的确定性筛选未实现（M2.5）。请求里的
  `temperature` / `top_p` 会被收下但不生效。按设计，抽样这一步永远在宿主侧
  ——引擎内不含任何随机数生成器。
- **慢**：Q8_0 上 prefill 64 token 约 6.6 s、decode 约 9 tok/s。纯标量反量化点积
  的必然结果，M5 用 SIMD + 融合反量化点积解决。
- **只支持 Qwen3 一个架构与一个模板**：刻意不引通用模板引擎（不引 Jinja）。
- **不做 NFC 归一化**：跟随 llama.cpp 的行为。
