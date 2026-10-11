# RPGBarEngine 设计

> 状态：**M3 已完成**（ChatML 渲染与官方模板逐字节一致；工具调用端到端打通）。
> 选型已定：**Qwen3-1.7B / Qwen3-4B**（2026-10-10 定案）。
> **随机性归属已定案**：引擎不产随机数（§12 #6）。
> 关联：RPGBar 仓库的 `docs/ENGINE-ARCHITECTURE.md`（Agent 与 Provider 层设计）

---

## 0. 当前进度

| 里程碑 | 状态 | 证据 |
| --- | --- | --- |
| M0 GGUF 解析 + 量化解码 + 分词器 | **完成** | `tools/run_checks.sh` 第 1–5 层全绿（见 §13） |
| M1 Qwen3 前向（标量） | **完成** | 3 个组合 × 29 项对拍全绿（见 §14） |
| M2 KV Cache + 贪心解码 | **完成** | 3 个组合逐 token 一致 + cache 自检逐位相同（见 §15） |
| M2.5 采样（温度 / top-p） | **未做**（等 §12 #6 的接口形状定案） | 引擎只出分布，随机数由宿主给 |
| M3 ChatML 模板 + tool_call 解析 | **完成** | 15 个用例逐**字节**与官方模板一致；工具调用端到端跑通（见 §16） |
| M4 C ABI 落地 | **完成** | 10 个函数走 dll 全通过：句柄生命周期、与工具链结果一致、前缀复用逐位等价（见 §17） |
| M5 SIMD 多线程 GEMM | 待开始 | — |
| M6 交叉编译 + 宿主打通 | 构建链已通 | `build/rpg_engine.dll` 零告警构建 |

---

## 1. 定位与目标

一个**进程内**的小型 LLM 推理引擎，编进 RPGBar 游戏本体，取代"玩家自己起 llama.cpp / Ollama"的部署方式。

| 目标 | 具体 |
| --- | --- |
| 零外部依赖 | 一个动态库 + 一份权重文件，不需要玩家装运行时、起服务、配端口 |
| 零 HTTP 往返 | 宿主直接函数调用 |
| 开箱可用 | 权重随包分发，装完游戏就能选到模型 |
| 双档位 | 小档服务玩家端，大档服务主机端（DM 需要更强） |
| **原生工具调用** | DM 靠多轮工具循环驱动状态机，这是硬需求（见 §2） |

非目标（本阶段不做）：GPU 后端、模型训练/微调、多模态、通用模型格式兼容。

---

## 2. 选型：为什么换族

### 2.1 硬约束：必须能 tool calling

RPGBar 的 DM 不是"写文章"，而是**驱动状态机**：`read_scene` → `roll_check` → `move_to` → `set_flag`，
`max_steps = 6` 的多轮工具循环（`RPGBar-Unity/server/agents/base.py`）。

宿主侧的调用前提是**模型返回结构化的 `tool_calls`**。**Gemma 2 没有原生工具调用模板** ——
在本地引擎上 `chat_with_tools` 必然失败，降级路径是"单轮 JSON"，此时 `tool_calls` 恒为空数组，
DM 只能输出文本、**改不了任何状态**。

所以：**工具调用是选型的第一约束，优先于中文质量、模型体积、许可。**

### 2.2 候选对比

| 族 | 小档 | 大档 | 原生 tool calling | 许可 | 引擎成本 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3** | **1.7B** | **4B** | ✔ Hermes 风格（`<tool_call>`） | **Apache-2.0** | 一套架构覆盖两档 |
| Qwen2.5 | 1.5B | 3B | ✔ 同族风格 | Apache-2.0（多数尺寸） | 同上，但档位偏小 |
| Llama 3.2 / 3.1 | 1B | 3B / 8B | ✔ | Llama 社区许可（含附加条款） | 3B 与 8B 架构参数不同，要验两遍 |

### 2.3 结论（建议，待确认）

**小档 Qwen3-1.7B，大档 Qwen3-4B。** 理由：

1. **档位正好对上**"小规模 2B / 大规模 4B"的要求（1.7B / 4B）；
2. **同族同架构** → 引擎实现一套前向即可覆盖两档，只是配置参数不同（层数 / 维度 / 头数）；
3. **原生工具调用**，且支持**非思考模式**（`enable_thinking=false`）—— 对固定 COT 任务很重要，
   否则模型会先输出 ` thinking…<｜end▁of▁thinking｜>` 拖慢速度、污染 JSON 解析；
4. **Apache-2.0 许可**：可自由商用与再分发，**不需要像 Gemma 那样传递使用限制条款** ——
   这条对 Steam 上线是实质性便利（Gemma 需在游戏内附 Gemma Terms 与使用限制通知）；
5. **体积小**：GGUF Q4_K_M 下 1.7B ≈ 1.1GB、4B ≈ 2.5GB。相比 Gemma 2 的 sfp（2B 就 2.98GB）**整体更小**，
   且"两个权重随包分发"因此变得可行。

### 2.4 量化

采用 **GGUF Q4_K_M**（llama.cpp 生态的标准 4-bit 混合量化）：

- 体积与质量的平衡点，生态验证充分；
- GGUF 规范公开、可自写解析器，不需要引入 llama.cpp 的构建体系；
- 保留升级到 Q5/Q8 或 fp16 的余地（同格式、不同量化类型）。

> **超参已实测确认**（不是假设）：官方 `config.json` + GGUF 元数据 + 张量清单三方
> 交叉核对，见 §13.2。
>
> | | Qwen3-1.7B | Qwen3-4B |
> | --- | --- | --- |
> | layers | 28 | 36 |
> | hidden | 2048 | 2560 |
> | Q heads | 16 | 32 |
> | KV heads | 8 | 8 |
> | head_dim | 128 | 128 |
> | intermediate | 6144 | 9728 |
> | vocab | 151936 | 151936 |
> | rope_theta | 1e6 | 1e6 |
> | rms_eps | 1e-6 | 1e-6 |
> | tied lm_head | 是（无 `output.weight`） | 是 |
> | 每层张量数 | 11 | 11 |

---

## 3. 架构分层

```
宿主：Unity C#（P/Invoke） / Python（ctypes）
        │
        ▼
  rpg_engine C ABI            include/rpg_engine.h —— 4 组函数，不暴露 C++ 类型
        │
        ├── Tokenizer        BBPE（Qwen 词表 151936）+ 特殊 token
        ├── Template         Qwen3 ChatML 渲染（含 tools）+ tool_call 解析
        ├── Model            权重常驻：GGUF 加载 → 量化张量 → 融合 GEMM
        ├── Session          KV Cache + 消息历史 + 采样参数（温度 / top-p）
        └── Kernels          RMSNorm / QK-norm / RoPE / SwiGLU / GEMM / Attention
```

**两级句柄**（见 §4）：`model` 是权重，`session` 是独立推理上下文。

---

## 4. C ABI

完整定义见 `include/rpg_engine.h`，这里只说设计取舍。

### 4.1 为什么是 C ABI

C++ 没有稳定 ABI —— 不同编译器、不同 STL 版本之间传 `std::string` / `std::vector` 会出事。
宿主是 Unity（IL2CPP / .NET）与 Python，都不该碰 C++ 类型。**四个函数的签名里只有 `int32_t` 和 `char*`。**

### 4.2 为什么 model 与 session 分开

这是本设计的关键一条：

- **一个进程只需要一个模型**（主机端加载 4B，玩家端加载 1.7B），权重加载贵、常驻内存；
- **但需要多个互相隔离的会话** —— 主机端的 **DM 与小助手共用同一份权重，上下文必须完全隔离**：
  DM 的会话里有全量剧本与 `dm_notes`，小助手的会话里只有公开视角（见 §8，对齐 §6 的"记忆隔离"原则）。

把这两件事揉成一个句柄，就只剩两种坏选择：要么让 DM 和小助手共用上下文（剧透 + 串味），
要么把权重加载两遍（内存翻倍）。**分开之后，两者都避免。**

### 4.3 会话与请求

- 调用方每轮把**完整 messages** 传进来，引擎负责与已有 KV Cache 做最长公共前缀匹配以复用；
- 需要彻底重来（例如换场景）时调 `rpg_engine_session_reset`；
- 返回的 `text` **已剥离思考块与工具调用块**，宿主拿到的就是"该说的话"；工具调用单独放 `tool_calls`。

---

## 5. Qwen3 前向要点

与 Gemma 2 的差异（**不是同一个网络**，代码无法照搬）：

| 项 | Gemma 2（TinyGemmaCpp） | Qwen3 |
| --- | --- | --- |
| 归一化 | RMSNorm，权重为 `1 + w`，pre + post 双归一化 | RMSNorm，权重为 `w`，pre-norm |
| 激活 | GeGLU（gelu-tanh） | **SwiGLU**（silu） |
| 注意力 | GQA + **sliding window**（每 2 层一个 global） | GQA，全因果，**无滑动窗口** |
| Q/K 归一化 | 无 | **QK-norm**（在 head_dim 上各做一次 RMSNorm） |
| 位置编码 | RoPE | RoPE（`rope_theta = 1e6`） |
| logit 处理 | 两处 soft-cap（50 / 30） | 无 |
| embedding | 乘 `sqrt(hidden)` | 无缩放，**tied**（小档 with lm_head 共享） |
| 权重格式 | `.sbs`（gemma.cpp 私有容器） | **GGUF** |
| 分词器 | SentencePiece BPE | **BBPE**（151936 词表） |
| 工具调用 | 无原生支持 | **原生**（`<tool_call>` 块） |

实现顺序上的两个容易踩的点：

1. **QK-norm 的位置**：在 RoPE **之前**、按 `head_dim` 归一化，每层都要做，漏掉就整体发散；
2. **tied embeddings**：`lm_head` 与 `token_embd` 共享权重，不要重复加载。

---

## 6. 工具调用协议

引擎不实现 Agent 逻辑，只做**渲染**与**解析**两件事：

```
宿主传入 tools（JSON Schema）
   │
   ├─ Template：按 Qwen3 的 ChatML 模板把工具定义渲染进 system 消息
   │
   ▼ 生成
   │
   ├─ Parser：从输出里提取 <tool_call>{"name":..., "arguments":{...}}</tool_call>
   │           （可能出现多个；非思考模式下不会夹杂  thinking 块，但仍要兼容跳过）
   │
   ▼
返回 {"text": "...", "tool_calls": [...]}
```

- 引擎**只解析、不执行** —— 工具执行、权限校验、状态写入全部留在宿主（对齐
  "护栏做在工具体里，不做在提示词里"）。
- 渲染的是 Qwen3 的**固定模板**，不引入 Jinja2 引擎（不做通用模板，只做这一个）。
- 强制 **non-thinking**：该模式下模型不输出思考块，对固定 COT 任务是纯收益。

---

## 7. 与 TinyGemmaCpp 的复用边界

**不共用代码库**，但把已验证的工程件手工搬过来（避免重复踩坑）：

| TinyGemmaCpp 模块 | 处理 |
| --- | --- |
| `src/matmul.cc`（829 行，AVX2/FMA 分块 GEMM） | **移植** —— 与模型无关，是性能主力 |
| `src/threads.cc`（线程池） | **移植** |
| `src/dtypes.h`（bf16/fp16 转换） | **移植** |
| `src/kv_cache.h` | **重写** —— 布局按 Qwen3 的 GQA 重算 |
| `src/sbs_reader.cc` | **换成 GGUF 解析**（新增约 300 行 + 量化类型解码） |
| `src/model.cc` / `ops.cc` | **重写** —— Gemma 算子（GeGLU / 双 soft-cap / PostNorm）不适用 |
| `tools/spm.py` | **不用** —— Qwen 是 BBPE，另写 tokenizer |
| `tools/verify_*.py`（对拍脚本） | **方法论沿用**（见 §11） |

---

## 8. 会话隔离（与 Agent 层的一致性）

对齐 `ENGINE-ARCHITECTURE.md` 的"记忆隔离"不变量：

| 会话 | 谁能看到 | 隔离要求 |
| --- | --- | --- |
| DM 会话 | 全量剧本 + `dm_notes` + 状态快照 | **不得**被小助手会话引用 |
| 小助手会话（每条任务一个） | 公开切片 + 小助手提示词 + 状态快照 | **不得**看到 `dm_notes` 或 DM 的对话历史 |

两个会话在引擎里是**两套独立的 KV Cache 与消息历史**，物理上不可能互相污染。
这既是防剧透，也是防"串味"导致的行为漂移。

> 主机端"小助手和 DM 不能共用记忆"这条，在引擎层的落点就是：**同权重、不同 session**。

---

## 9. 性能目标

参考基准：TinyGemmaCpp 的 Gemma 2 2B 在 i7-14650HX（16 线程）上实测 decode **21.4 tok/s**。

| 场景 | 目标 | 说明 |
| --- | --- | --- |
| Qwen3-1.7B decode | **≥ 15 tok/s** | 玩家端"建议"约 200 token → **< 15s**，可接受 |
| Qwen3-1.7B prefill | ≥ 100 tok/s | 300 token 上下文 < 3s |
| Qwen3-4B decode | **≥ 7 tok/s** | 主机端 DM 一轮约 250 token → **25–35s** |
| Qwen3-4B prefill | ≥ 50 tok/s | — |

**必须诚实记下的体验代价**：4B 在 CPU 上**每轮要几十秒** —— 这是"DM 走本地引擎"的固有代价，
不是实现缺陷。缓解方向（按优先级）：更激进量化（Q4 → Q3）、更多线程、后续 GN 后端。
若体验不可接受，DM 仍可切回云 API（Provider 层支持，见 `ENGINE-ARCHITECTURE.md` §4）。

---

## 10. 里程碑

每一步的终点都必须是"可运行、可验证"，不留半成品。

| # | 内容 | 完成判据 |
| --- | --- | --- |
| **M0** | GGUF 解析 + BBPE tokenizer | 对一段已知文本，tokenizer 与官方逐 id 一致 |
| **M1** | Qwen3 前向（无 KV Cache，标量） | 与 PyTorch 官方实现逐层对拍：**逐元素相对误差 p99 ≤ 2e-4，且每个位置的 argmax 一致**（判据修正的经过见 §14.2） |
| **M2** | KV Cache + 贪心解码 | 贪心解码 N 个 token 与官方**逐 token** 一致（**已通过**，见 §15） |
| **M2.5** | 采样（温度 / top-p） | 接口形状待定（§12 #6）。抽样**不在引擎内**，引擎只出确定性分布 |
| **M3** | ChatML 模板 + tool_call 解析 | 给定工具定义，能稳定产出结构化 `tool_calls`（**已通过**，见 §16） |
| **M4** | C ABI + 多会话 | 同一模型两个会话互不干扰；前缀复用与冷启动结果一致（**已通过**，见 §17） |
| **M5** | SIMD / 多线程 GEMM（移植 TinyGemmaCpp） | 达到 §9 性能目标 |
| **M6** | 交叉编译出 `rpg_engine.dll`，Unity P/Invoke 打通 | Unity 里调用成功生成一段合法 JSON |

---

## 11. 验证方法

沿用 TinyGemmaCpp 已验证的分层对拍（每层都留可复现脚本），现在有九层：

| 层 | 对拍对象 | 判据 | 对应脚本 |
| --- | --- | --- | --- |
| 1 进程内自检 | 自己（不变量） | 108 项全过 | `selftest.cc` |
| 2 量化布局 | GGUF 写入器算出的 offset | 块尺寸与张量铺满推算一致 | `check_quant_layout.py` |
| 3 量化语义 | `gguf.quants`（llama.cpp 官方算法） | **逐位**相同 | `xcheck_dequant.py` |
| 4 分词 | 官方 tokenizer | 逐 **id** 一致 | `tok_ref.py` |
| 5 跨量化档位 | Q8_0 与 Q4_K_M 的同一张量 | 高相关；F32 张量逐位相同 | `xcheck_quant.py` |
| 6 前向（M1） | PyTorch 官方 `Qwen3ForCausalLM` | **相对误差 p99 ≤ 2e-4 且 argmax 一致** | `check_forward.py` |
| 7 生成（M2） | PyTorch 官方 `Qwen3ForCausalLM` | **逐 token** 相同（比第 6 层更硬） | `check_generate.py` |
| 8 ChatML 渲染（M3） | HF 官方 `apply_chat_template` 跑官方模板原文 | 逐**字节**相同 | `check_chatml.py` |
| 9 C ABI（M4） | 工具链路径 + 冷启动会话 | 结果完全相同；句柄生命周期正确 | `check_abi.py` |

M3 另有一条端到端验收（不在分层表里，因为它要跨好几层）：`check_toolcall.py`
—— 渲染带工具定义的 prompt → 引擎贪心生成 → 解码 → 解析出结构化 `tool_calls`，
同时与官方实现逐 token 对拍。

**为什么第 9 层的参照物是"自己"**：C ABI 这一层没有外部参考物可对（宿主就是我
自己写的脚本），能对的是**另一条路径**：同一份请求分别走"命令行工具链"和"dll
入口"，产出必须完全一致。另外它自带一条独立判据 —— 暖缓存（同一会话续写）与
冷缓存（新会话）结果必须逐字节相同。前缀复用是整个 ABI 里最容易静默出错的地方，
而复用错不报错、只让模型看到一段错的上下文。

**为什么第 8 层是"逐字节"而不是"语义等价"**：模板里的空格、`\n` 数量、空思考块
的有无，都会影响模型的行为；"看起来差不多"的渲染没法解释"为什么输出不一样了"。
参考侧用的是官方 `tokenizer_config.json` 里的 `chat_template` **原文**，所以比的
对象是模板本身，不是"某个人对模板的理解"。

**一个由此暴露的地基问题**：我们的 JSON 序列化原本是紧凑格式（`{"a":1}`）且浮点
用 `%.17g`（0.7 → 0.69999999999999996）。而官方模板用 Jinja 的 `tojson`，输出是
Python `json.dumps` 的默认样式（`{"a": 1}` 且用最短往返表示）。**这条对拍一开始
就红了** —— 正是它逼出了 `Json::DumpPython()` 与最短往返浮点格式化（见 §16.3）。

**为什么第 6 层是"相对误差"而不是"绝对误差"**：见 §14.2 —— 残差流里有一个
量级 1.2e4 的离群维度，在那里 float32 的 1 ULP ≈ 1e-3，绝对判据失去意义。

**为什么第 7 层比第 6 层硬**：第 6 层只要求"分布差得足够小"，一个 1e-6 的偏差
只要不改变 argmax 就算过；第 7 层要求**每一步的 argmax 都对** —— 错一步，
后面全部跟着变（自回归会把误差放大）。而且它顺带把 KV Cache 也验了（跑的是
增量路径）。

**第 7 层还内置一条自检**：`gen_check` 会把"带 cache 的 prefill"与"不带 cache
的全序列"的 logits **逐位**比一次。这条能一次性证明 cache 的写入/读取没把布局
搞错 —— 布局错了不会崩，只会让生成结果慢慢跑偏，那种最难查。

**为什么第 3 层必须逐位**：量化解码有损，一旦解码有偏差，上层的"误差很小"
可能是错的巧合。只有先钉死"同一份量化字节解出来同一个数"，上层的数值差异
才可归因。实测这条已经成立（含 3.1 亿元素的整张词表，逐位相同）。

---

## 12. 已定事项

| # | 问题 | 结论 |
| --- | --- | --- |
| 1 | **模型族** | **Qwen3**（2026-10-10 定案）：1.7B / 4B 正好对上"小规模 2B / 大规模 4B"；Apache-2.0 可自由再分发；原生 tool calling；可控思考模式 |
| 2 | 量化档位 | 目标 **Q4_K_M**；对拍基准用官方 **Q8_0**（几乎无损，能把"量化误差"和"实现错误"分开） |
| 3 | 上下文长度上限 | 暂定 4096，实测后调 |
| 4 | 权重是否随包 | 两档随包（Q4_K_M 1.03 GiB + 4B 约 2.5 GiB），首启校验 |
| 5 | 引擎是否支持多模型族 | **先只做 Qwen3**；但架构不写死（配置全从 GGUF 读），后续要扩族是"加一个前向 + 一套模板"，不是重写 |
| 6 | **随机性归属**（2026-10-10 定案） | **引擎不产随机性**。温度 / top-p / top-k 是**宿主侧参数**：引擎只做确定性的"算分布 + 筛选"，"按概率抽一个"由宿主完成（宿主传一个 `[0,1)`，引擎按累积概率做逆变换）。因此**引擎内没有 RNG**，同一输入必得同一输出。三条理由：① 对齐 RPGBar 的"骰子只有一个来源" —— 模型/引擎只产出权重或分布，从不自己掷骰；② 随机流集中在宿主侧，**整局对局可复现**（同种子 + 同输入 → 同剧情），且多会话（DM / 小助手）不会互相扰动随机流；③ 引擎是**纯函数**，最好对拍、最好排错。玩家侧"创造力开关" = 宿主配置里的 `temperature`，DM 与小助手各自可调 |
| 7 | 采样是否属于引擎 | **不属于**。引擎只提供 `sample_probs`（温度/截断/重归一化，纯函数）。M2 阶段先只做**贪心**（零随机数） |

---

## 13. M0 落地记录（已完成）

### 13.1 交付物

```
include/rpg_engine.h      C ABI（10 个函数，两级句柄）
src/
  common.h/.cc            错误（异常 → ABI 边界翻译）、日志
  bytes.h                 小端读取游标、对齐
  dtypes.h                f32/f16/bf16 互转（RNE 舍入）
  unicode.h                UTF-8 编解码、\p{L}/\p{N}/\s、GPT-2 字节⇄码点
  unicode_tables.h        **生成物**（tools/gen_unicode_tables.py）
  json.h/.cc              最小 JSON（解析/序列化，零依赖）
  mmap_file.h/.cc         跨平台只读 mmap（Windows/POSIX）
  threads.h/.cc           物理核探测 + 自研常驻线程池
  quant.h/.cc             ggml 类型表 + 15 种类型的反量化
  gguf.h/.cc              GGUF v1/v2/v3 解析（mmap 零拷贝）
  tokenizer.h/.cc         Qwen BBPE（特殊 token 切分 + 手写预分词正则 + BPE 合并）
  rpg_engine.def          Windows 导出清单
  engine_stub.cc          C ABI 占位（M4 替换）
tools/
  dump_gguf.cc            检视 GGUF + 抽样反量化
  tok_check.cc            分词对拍协议进程（NUL 分隔）
  selftest.cc             69 项进程内自检
  gen_unicode_tables.py   生成 Unicode 范围表
  check_quant_layout.py   用文件 offset 独立验证块尺寸
  tok_ref.py              HF tokenizers 对拍
  xcheck_quant.py         跨量化档位交叉校验
  run_checks.sh           一条命令跑完所有检查
scripts/fetch_model.py    权重与对拍参考物下载（断点续传，走 hf-mirror）
```

### 13.2 已确认的事实（不是假设）

从官方 `config.json`、GGUF 元数据、张量清单三方核对：

- Qwen3 的 GGUF 架构键前缀是 `qwen3.`；`head_dim` 单独落在
  `qwen3.attention.key_length`（**不等于** `hidden / heads`，必须单独读）。
- `tokenizer.ggml.pre = "qwen2"` → 预分词走 LLAMA_VOCAB_PRE_TYPE_QWEN2 的模式。
- `tokenizer.ggml.model = "gpt2"` → byte-level BPE，merges 共 151387 条。
- **无 `output.weight`** → tied embeddings，lm_head 复用 `token_embd.weight`。
- 每层 11 个张量：`attn_{q,k,v,output,q_norm,k_norm}`、`attn_norm`、
  `ffn_{gate,up,down}`、`ffn_norm`。
- GGUF 的 `dims[0]` 是最快维，所以权重在内存里是行主 `[rows=dims[1], cols=dims[0]]`
  —— 正好是 `out = x · Wᵀ` 需要的布局。
- 张量类型分布能直接看出档位：Q8_0 文件是 `F32×113 + Q8_0×197`；
  Q4_K_M 是 `F32×113 + Q4_K×168 + Q6_K×29`（K-quant 混合量化的典型形状）。

### 13.3 M0 阶段做的几个设计决定

1. **并行后端不用 OpenMP，自研常驻线程池。**
   这个库要直接丢进 Unity 游戏。libgomp / vcomp 是额外的运行时依赖，
   缺了就是"加载失败"，宿主自己用 OpenMP 时还会撞车。而且 decode 是
   "很多次很小的并行区"，OpenMP 每轮 fork/join 的开销在此不可忽略。

2. **权重走 mmap，不整份读进堆。**
   1.8 GB 读进 `vector` 会让峰值内存翻倍（页缓存 + 副本）。mmap 之后张量数据
   就是文件页，未触碰的层不占物理内存。

3. **不引正则库，手写 qwen2 预分词。**
   那个模式只用一个分支、几个字符类和一个负向前瞻，但通用正则引擎要支持
   Unicode 类别与回溯语义，代价远大于收益。手写版本按分支顺序与贪婪/回溯行为
   逐条对齐 Rust regex 的"最左优先"语义，再用 3000 条随机文本对拍验证。

4. **Unicode 类别用生成表，不用手写近似。**
   手写"ASCII + CJK 区间"在韩文、希腊字母、全角数字上会静默切错。范围表由
   `tools/gen_unicode_tables.py` 从 UCD 扫全码位生成（约 5 KB），可复现。

5. **`%zu` 全项目禁用。**
   MinGW 的 printf 走 MSVCRT，不支持 `%zu`，会静默输出乱码。统一用
   `RPG_SZ` / `RPG_SZ_ARG` 宏（展开成 `%llu` + 转换）。

6. **Windows 导出用显式 `.def`。**
   不给 `.def` 时 MinGW ld 会导出**全部**符号 —— 实测有 200+ 条，含内部 C++
   修饰名与 libgcc 的 `_Unwind_*`。现在导出表恰好 10 个 C 函数。

### 13.4 已知差异（刻意保留）

**不做 NFC 归一化。** Qwen3 的 `tokenizer.json` 里声明了 `normalizer: NFC`，
但 llama.cpp 对 BPE 不做任何 Unicode 归一化。我们的端到端基准是 llama.cpp，
所以选择与它一致：

- 输入已是 NFC（绝大多数中英文本、以及我们自己的 chat 模板输出）→
  与 HF tokenizers **完全一致**；
- 输入是分解形式（如 `e` + U+0301）→ 会与 HF 不同。

要补的话需要完整的 NFC 表与"重排 + 组合"算法，属于独立工作。
当前对拍脚本把所有语料先归一成 NFC，保证比较的是同一件事。

### 13.5 M0 的验证结果（三层，逐条可复现）

| 层 | 手段 | 结果 |
| --- | --- | --- |
| 进程内自检 | `build/selftest.exe` | **69/69**（字节读写、UTF-8、字节⇄码点双射、Unicode 类别、JSON、简单量化） |
| 量化块尺寸 | `check_quant_layout.py`，用 GGUF 写入器算出的 offset 反推 | Q8_0 1.0625、Q4_K 0.5625、Q6_K 0.8203125、Q2_K 0.328125、Q3_K 0.4296875、Q5_K 0.6875 **全部与实现表一致**；四个文件的张量无缝铺满，推算总长与文件大小**逐字节相同** |
| 量化语义 | `xcheck_quant.py`：Q8_0 与 Q4_K_M 同一张量 | 相关系数 **0.9966–0.9998**（符合 4-bit 量化噪声）；F32 张量**逐位相同** |
| 分词 | `tok_ref.py` 对 82 条构造用例 + 3000 条随机模糊用例 | **3031/3031 逐 id 一致**（HF `tokenizers` 为参考） |

> 为什么"量化块尺寸"要用文件 offset 来验：这张表**写错了不会报错**，
> 只会读错字节，最后表现为模型胡说八道。而 GGUF 的 offset 是写入器按它自己的
> 尺寸表算的，与我们的实现完全独立 —— 两者一致，才算把这张表钉死。

### 13.6 下一步（M1）

已于本轮完成，落地记录见 **§14**。

> 原计划：标量 Qwen3 前向 → 与 PyTorch 参考对拍（逐张量误差 < 1e-4）。
> 实际执行时发现"绝对 1e-4"在 float32 下不可达，判据经过修正 —— 见 §14.2。


---

## 14. M1 落地记录（已完成）

### 14.1 交付物

```
src/kernels.h            标量算子：Dot / RmsNorm / Softmax / SiLU / RoPE（M5 换 SIMD 的边界）
src/model.h/.cc          ModelConfig + QuantMatrix（量化矩阵视图）+ Model::Load（形状全量校验）
src/qwen3.h/.cc          ForwardScratch + ForwardLayer + CausalAttention + ForwardFull
tools/npy.h              极小 .npy 写出器（只给 tools 用，避免为对拍引依赖）
tools/forward_check.cc   跑一次前向，导出 tokens / logits / 逐层残差流为 npy
tools/qwen3_ref.py       参考侧共用模块：GGUF → 官方 Qwen3ForCausalLM（含精度补丁开关）
tools/ref_forward.py     参考前向（支持 --layers / --dtype / --norm-identity）
tools/cmp_npy.py         逐层比较（相对误差分位数），可 import 复用
tools/check_forward.py   M1 一次性跑完所有组合
tools/check_noise_floor.py  float32 噪声底诊断（判据的依据）
tools/xcheck_dequant.py  反量化逐位对拍（vs gguf.quants）
tests/prompts/zh_chatml.txt  中文 ChatML 长提示词
```

`Model` 的加载期校验值得单说：张量名、维数、形状、类型全部逐条对，任何一条
不符立刻抛错并打印"期望 [行,列] / 实际 [行,列]"。**不做"缺了就用默认值兜底"**——
兜底出来的错误会推迟到几十层之后以 NaN 的形式出现，那时线索全没了。

### 14.2 判据的修正（这是 M1 最重要的一条记录）

**原判据**：逐 tensor 误差 < 1e-4。**实测**：逐层残差流的 `max|Δ| = 0.0117`，
"超限 100 倍"。但同一份结果的 top-5 token 与参考**完全一致**。

看一眼量级就明白了：

| 项 | 值 |
| --- | --- |
| 残差流里有离群维度 | `|h| ≈ 1.19e4`（第 2 层输出出现，之后一直保持） |
| 该量级下 float32 的 1 ULP | `0.0009765625` |
| 观测到的 max\|Δ\| | `0.011719` = **恰好 12 ULP** |

也就是说，在 `|h| = 1.2e4` 上，"绝对 1e-4"比一个 ULP 还小 10 倍 ——
**这个判据在 float32 下不可能满足**，与实现好坏无关。这是判据写错了，
不是实现错了。

改成**逐元素相对误差的分位数**（只统计 `|ref| > 1e-3` 的元素，剔除近零坐标
造成的相对误差失真），判据 `p99 ≤ 2e-4`。绝对 max|Δ| 仍然打印，但不作判据：
它被两类元素主导而失去信息量 —— 巨值维度（ULP 大）与近零坐标（绝对差极小）。

**"2e-4 这个数是怎么来的"** —— 不是拍的，是量出来的（§14.3）。

### 14.3 精度证据链

**第一环：反量化逐位相同。** `xcheck_dequant.py` 把我们的整张张量反量化结果与
`gguf.quants`（llama.cpp 官方算法的 Python 移植）逐位比：

| 张量 | 形状 | 类型 | 结果 |
| --- | --- | --- | --- |
| `blk.0.attn_norm.weight` | (2048,) | F32 | 逐位相同 |
| `blk.0.attn_q.weight` | (2048, 2048) | Q8_0 | 逐位相同 |
| `blk.0.ffn_gate.weight` | (6144, 2048) | Q8_0 | 逐位相同 |
| `blk.0.ffn_down.weight` | (2048, 6144) | Q8_0 | 逐位相同 |
| `token_embd.weight` | (151936, 2048) | Q8_0 | 逐位相同（3.1 亿元素） |
| `blk.13.attn_v.weight` | (1024, 2048) | Q8_0 | 逐位相同 |
| Q4_K_M 版 | 同上四张 | **F32 / Q4_K / Q6_K** | 逐位相同 |

这条排除了"权重本身有偏差"这一类原因 —— 于是上层的差异只可能来自前向的
浮点运算顺序。

**第二环：量出 float32 的噪声底。** `check_noise_floor.py` 把三个东西放在
**同一深度**上比：`ours32`（我们）、`ref32`（官方实现，float32）、
`ref64`（官方实现，float64 ≈ 真值）。其中 ref64 用 `use_exact_rmsnorm()` 打补丁 ——
官方 `Qwen3RMSNorm` 内部强制把输入降到 float32 再算，不拿掉这个舍入，
"float64 真值"就不真。

12 层深度下的 p99 相对误差：

| | 我们 | 噪声底（ref32 vs ref64） | 比值 |
| --- | --- | --- | --- |
| embedding | 0 | 0 | — |
| layer 0 | 4.1e-05 | 3.69e-05 | 1.11 |
| layer 5 | 6.36e-05 | 5.10e-05 | 1.25 |
| layer 11 | 9.03e-05 | 6.95e-05 | 1.30 |

**比值 1.06–1.40**：我们的相对误差与"float32 + 这套算子"能做到的最好水平
同量级。这就是"实现没有算法问题"的量化依据，也是 `2e-4` 判据的来源
（噪声底在 28 层上会到 ~1.5e-4，留一倍余量）。

**第三环：单层可测。** layer 0（一个完整的 transformer 块：pre-norm →
QKV → QK-norm → RoPE → GQA 因果注意力 → o_proj → pre-norm → SwiGLU）
在显著元素上的相对误差中位数是 `8.3e-7`。任何一处结构性错误（相邻配对
RoPE、漏掉 QK-norm、GQA 分组错、漏 silu、多乘 `sqrt(hidden)`）都会让这里是
O(1) 而不是 O(1e-6)。

**一个副作用改进**：`Dot` / `DotRowQuant` 原本用单个串行累加器。单串行求和的
舍入误差随长度**线性**增长，而 BLAS 的分块/向量化求和是对数级 —— k 到 6144 时
实测差 4–5 倍。改成 4 个独立部分和后，layer 0 的绝对误差从 `1.43e-5` 降到
`3.81e-6`（比值 5.0 → 1.33）。4 个累加器正好对应 M5 里 SIMD 的 4 个 lane，
所以这个结构可以直接留用。

### 14.4 验证结果

`python tools/check_forward.py`，3 个组合 × 每组合 29 项（embedding + 28 层
+ logits）：

| 组合 | 提示词 | 序列 | 最差 p99 | argmax | 结果 |
| --- | --- | --- | --- | --- | --- |
| q8_ascii | `The capital of France is` | 5 | 1.47e-4 | 5/5 一致 | PASS |
| q8_zh | 中文 ChatML（含特殊 token） | 64 | 1.38e-4 | 64/64 一致 | PASS |
| q4_zh | 同上，换 Q4_K_M 权重 | 64 | 1.34e-4 | 64/64 一致 | PASS |

其它可观察的旁证：

- `The capital of France is` → top-1 ` Paris`（logit 22.92）
- 中文 ChatML 提示词 → top-1 `<think>`（logit 30.61）—— **正是 Qwen3 默认
  思考模式该有的行为**，说明模板与特殊 token 的正确性在引擎侧已经成立
- 中文提示词经分词器往返**逐字节还原**
- logits 的 `max|Δ|` 只有 `9.25e-05` —— 连原来那个绝对 1e-4 判据都过了
  （它过只是因为 logits 这一层的量级小，不代表逐层残差流也能过）

### 14.5 已知差异与限制

- **fp32 精度**：内部全程 float32。误差水平见 §14.3。提高精度的路径（若需要）
  是 Kahan 补偿求和，代价约 4 倍算力 —— 当前没有必要。
- **`--layers N` 截断时不产出 logits**：截断模式下末端归一化与 lm_head 依赖更深的
  层，返回空 logits 而不是一个"看起来能用"的错值。
- **单线程 attention**：GEMM 按输出行并行，attention 走标量。这是刻意的 ——
  并行会改浮点累加顺序，M1 阶段的首要目标是可对拍。
- **prefill 性能还很差**：Q8_0 上 64 token 用 9.2 s（约 7 tok/s），
  离 §9 的 100 tok/s 目标差一个数量级。这是纯标量 `DotRowQuant`
  （每元素一次反量化）的必然结果，M5 用 SIMD + 融合反量化点积解决。

### 14.6 下一步（M2）

**M2 已完成**，落地记录见 §15。当时的计划是：

1. KV Cache，布局 `[n_layers, n_kv_heads, max_seq, head_dim]`，K 与 V 分开
   两块连续内存；`ForwardLayer` 换成带 cache 的版本。
2. 采样：**M2 只做贪心**（零随机数，判据可硬对拍）。温度 / top-p 的确定性
   筛选是独立的一步（见 §12 #6）：引擎算出候选分布，抽样由宿主给一个 `[0,1)`。
   引擎内**不设 RNG** —— 随机性全部集中在宿主侧，引擎只保留确定性的部分（纯函数）。
   （本节最初写的"引擎只接受一个 seed"是错的，与同句前半段自相矛盾，已改掉）
3. 判据：贪心解码 N 个 token 与官方逐 token 一致。

三条都按计划做到了（第 1 条的实现比原计划更好：没有另写一套增量代码，而是让
同一个 `ForwardLayer` 同时支撑两条路径，于是"填 cache"与"不填 cache"可以互相
印证）。

---

## 15. M2 落地记录（已完成）

### 15.1 交付物

```
src/kv_cache.h/.cc       KvCache：布局 [n_layers, n_kv_heads, max_seq, head_dim]
src/qwen3.h/.cc          + ForwardStep（单 token 增量）、GenerateGreedy（贪心）
                         + AttentionFromCache（从 cache 读历史的因果注意力）
src/kernels.h            + ArgMax（并列取最小下标，与 torch.argmax 一致）
tools/gen_check.cc       生成探针：prefill → 贪心生成 → 写盘 + cache 自检
tools/cli_util.h         工具公共件（id 列表解析 / 按字节读写 / 二进制 stdio）
tools/check_generate.py  M2 一次性跑完所有组合（逐 token 比）
tools/run_checks.sh      扩到七层
```

### 15.2 关键实现决定：**不写第二套前向**

原计划是"另写一个增量版 attention"。实际做法是让**同一个 `ForwardLayer` 同时
支撑两条路径**，靠三个参数区分：

| 场景 | seq | pos0 | kv |
| --- | --- | --- | --- |
| 全序列（M1） | N | 0 | nullptr |
| prefill | N | cache.len() | 非空 |
| 增量解码 | 1 | cache.len() | 非空 |

这么切带来三个好处：

1. **两条路径能互相印证**。同一段 prompt，"填 cache 的 prefill"与"不填 cache 的
   全序列"走的是同一份代码、同一套累加顺序，所以 logits 必须**逐位相同**。
   这条断言被 `gen_check` 当自检跑：实测 759680 与 9723904 个数值**全部逐位相同**。
   cache 布局写错（头与位置顺序颠倒、stride 算错……）不会崩，只会让生成结果
   慢慢跑偏 —— 那种 bug 极难查，而这条自检把它变成一次断言。
2. **不会出现"两条路径行为不一致"**。另写一份增量 attention 的话，任何一处
   QK-norm / RoPE 的位置差异都会变成"prefill 与 decode 对不上"，而且归因困难。
3. 增量只是 seq=1 的特例，不需要额外维护一套代码。

### 15.3 一个差点漏掉的不变量

`GenerateGreedy` 第一版在生成最后一个 token 后**没有**再走一步前向（理由是
"不需要下一个 token 的 logits"），于是 cache 长度成了 `prompt + N - 1`。

这不是崩溃，是更糟的东西：**短一个**。下一轮对话在 cache 上续写时会缺一个位置
的 K/V，模型看到的历史比实际少一个 token —— 不报错，只让输出悄悄变味。

已改成：**每个返回的 token 都写进 cache，包括最后一个与停止 token**。代价是末尾
多算一步前向（约 0.1 秒），换来一条可一句话验证的不变量：

```
cache.len() == prompt 长度 + 返回的 token 数
```

`gen_check` 会打印它（实测 5 + 16 = 21）。

### 15.4 验证结果

`python tools/check_generate.py`，判据"逐 token 相同"：

| 组合 | 提示词 | prompt | 生成 | 结果 |
| --- | --- | --- | --- | --- |
| q8_ascii | `The capital of France is` | 5 | 24 | **24/24 一致** |
| q8_zh | 中文 ChatML（含特殊 token） | 64 | 24 | **24/24 一致** |
| q4_zh | 同上，换 Q4_K_M 权重 | 64 | 24 | **24/24 一致** |

KV Cache 自检（带 cache vs 不带 cache 的 prefill，逐位比）：

| 组合 | 比较的数值个数 | 结果 |
| --- | --- | --- |
| q8_ascii | 759680 | 逐位相同 |
| q8_zh | 9723904 | 逐位相同 |

旁证：`The capital of France is` 诚实往下续写成
`Paris. The capital of the United States is Washington, D.C. The capital of the
United Kingdom is London. The` —— 24 步里没有跑偏、没有重复。

### 15.5 已知限制

- **decode 约 9 tok/s**（Q8_0，单 token 前向约 108 ms）。这是纯标量
  `DotRowQuant` 的必然结果（读每个权重元素都要现反量化一次），M5 用
  SIMD + 融合反量化点积解决，目标 §9 的 100 tok/s。
- **prefill 仍是整段全序列**，没有分块，64 token 约 6.6 s。M5 一并解决。
- **attention 仍走标量、单线程**。按 head 并行不改每个输出的累加顺序（可以并行），
  但当前瓶颈不在这里，留到 M5 与 GEMM 一起做。
- **cache 只支持顺序追加**。多轮对话靠"前缀复用"：`ForwardFull` 已支持从
  `cache.len()` 继续 prefill，但"算出与上一轮的最长公共前缀"由宿主负责（M3/M4 落）。
- **上下文满了明确报错**，不静默截断 —— 静默停下会让人以为"模型自己讲完了"，
  而实际是宿主把 `max_seq` 配小了。
- **采样（温度 / top-p）还没有代码**。M2 只做了贪心。见 §12 #6。

### 15.6 下一步

1. ~~**M3 ChatML 模板 + tool_call 解析**~~ —— **已完成**，见 §16。
   （本节最初写的"空 ` thinking\n\n\n\n`"是**错的**：实测官方模板里的空思考块是
   `<think>\n\n</think>\n\n`，且特殊 token 就叫 `<think>` / `</think>`，无前导空格。
   凭记忆写必然出错 —— 所以 §16 把"参考物是官方模板原文"这件事放在最前面。）
2. **M2.5 采样**：接口形状按 §12 #6 —— 引擎出确定性分布，随机数由宿主给。
   现在 M3 已通，采样是下一个"想做再做"的项：DM 生成要的是**稳定的结构化输出**
   （温度 0 就够），而工具调用是硬需求；"创造力"这个旋钮是给想调的人调的，
   不是主路径。**M4（C ABI）优先于它** —— 宿主现在就差一个能调用的入口。

---

## 16. M3 落地记录（已完成）

### 16.1 交付物

```
src/chat_template.h/.cc   Qwen3 ChatML 渲染 + messages/tools 的 JSON 解析
src/tool_parser.h/.cc     输出解析：剥控制标记 / 思考块 / 抽 <tool_call>
src/json.h/.cc            + DumpPython（Python 的 json.dumps 样式）
                          + 数字保留整数字面量、浮点用最短往返表示
tools/chatml_check.cc     渲染一份 case 并落盘
tools/chatml_ref.py       参考侧：HF apply_chat_template 跑官方模板原文
tools/check_chatml.py     第 8 层：逐【字节】对拍（15 个用例）
tools/parse_check.cc      把一段输出文本喂进解析器，打印结构化结果
tools/check_toolcall.py   M3 端到端验收：工具定义 → 生成 → 结构化 tool_calls
tests/chatml/dm_toolcall.case.json   带 2 个工具定义的 DM 场景
```

### 16.2 参考物是**官方模板原文**，不是"我们对模板的理解"

`scripts/fetch_model.py` 的 REFS 里加了 `tokenizer_config.json` —— 它的
`chat_template` 字段就是 Qwen3 的官方 Jinja 模板，参考侧直接把它交给
transformers 的渲染器跑（`PreTrainedTokenizerFast.chat_template = 原文`），
不重新实现任何东西。于是对拍的对象是模板本身。

**这一步不是多余的：凭记忆写必错。** 两处实测才知道的细节：

| 我以为 | 实际 |
| --- | --- |
| 空思考块是 ` thinking\n\n\n\n` | **`<think>\n\n</think>\n\n`**，特殊 token 就叫 `<think>`(151667) / `</think>`(151668)，**没有前导空格** |
| 只有"最后一条 assistant"才带思考块 | 条件是 `loop.index0 > last_query_index`，其中 `last_query_index` 是从后往前找的第一条"真实用户提问"（role=user 且内容不是整条 `<tool_response>` 包裹） |

模板里还有两处反直觉、但错了不崩只是结果不同的分支，都在 `chat_template.cc`
的注释里点明了：**连续的 `tool` 消息合并成一条 `<|im_start|>user`**；**tools
分支下 messages[0] 的 system 不会在循环里重复渲染**。

### 16.3 一个被对拍逼出来的地基问题

第一次跑第 8 层，**`12_tricky_tool_json` 直接红了** —— 差异全在 JSON 序列化上：

| 项 | 我们原来 | 官方 `tojson`（= Python `json.dumps`） |
| --- | --- | --- |
| 分隔符 | `{"a":1}` | `{"a": 1}`（`, ` 与 `: ` 后带空格） |
| 浮点 | `%.17g` → `0.10000000000000001` | 最短往返 → `0.1` |
| 整数/浮点 | 一律当 double，`1.0` 输出成 `1` | `1` 与 `1.0` 各保持原样 |

改了三处：`Json::DumpPython()`（新样式）、`Json::IsIntLiteral()`（解析时记住
字面写法）、`ShortestFloat()`（从 1 位有效数字往上试到能往返）。

**这不只是"为了对拍"**：M4 的 C ABI 要把 `temperature: 0.7` 回写给宿主，
`%.17g` 会输出 `0.69999999999999996` —— 合法但谁看了都别扭。

测试用例里刻意用 `0.1` 和 `1e-05` 而不是 `0.5`：后者两种写法结果一样，测不出
区别。这种"挑得出问题的输入"比"看起来像回事的输入"值钱。

### 16.4 解析器的契约：**必须保留特殊 token**

M3 端到端第一次跑也红了，而且红得有价值 —— **`<tool_call>` / `</tool_call>`
自己就是特殊 token**（151657 / 151658）。解码时开 `skip_special` 会把这两个
标记一起删掉，解析器只能看到裸 JSON，还会把 JSON 当成旁白念给玩家听。

所以契约定成：**喂给解析器的是 `skip_special = false` 的解码文本**，"哪些标记
不该给玩家看"由解析器的 `StripControlMarkers` 负责（`<|im_start|>` 连同紧跟的
role 词、`<|im_end|>`、`<|endoftext|>`）。名单里刻意**不含** `<tool_call>` 与
`<think>` —— 那两个是结构标记，得先被结构扫描消费掉。

顺带得到一个好处：**宿主有没有配停止 token 都不影响解析**。M2 的
`GenerateGreedy` 会把手上的停止 token 也放进返回序列（对齐 HF 行为），以前这
意味着文本尾部会拖一个 `<|im_end|>`；现在解析器顺手清掉了。

三个降级路径（都设了标记，宿主可以据此决定重试还是当失败）：

| 情况 | 行为 |
| --- | --- |
| `<think>` 没闭合 | 该处到结尾都算思考内容，`unterminated_think = true` |
| `<tool_call>` 没闭合 | 仍尝试解析并交出结果，`unterminated_tool_call = true` |
| `arguments` 不是合法 JSON | 该调用 `ok = false`，`arguments_raw` 保留原文供宿主排查 |

另外 `arguments` 是字符串时会再拆一层（有的模型会把参数序列化成字符串塞进去）。

### 16.5 验证结果

**第 8 层（逐字节，`python tools/check_chatml.py`）：15/15 全绿。**
覆盖：有/无 tools、有/无 system、assistant 显式 `reasoning_content`、
content 内嵌 `</think>`、`last_query_index` 前后、连续 tool 响应合并、
多条 tool_calls、生成提示开关、思考模式开关、工具定义里的刁钻值
（浮点 / 布尔 / null / 嵌套数组 / 转义 / 中文）。

**进程内自检：106/106**（新增 37 项：JSON Python 样式、ChatML 渲染、输出解析）。

**端到端（`python tools/check_toolcall.py`）** —— 这就是 M3 的完成判据：

| 项 | 结果 |
| --- | --- |
| 渲染（2 个工具定义） | 1251 字节 ChatML prompt |
| 引擎贪心生成 | 22 个 token（遇到 `<|im_end|>` 停） |
| 与官方实现逐 token 对拍 | **22/22 一致** |
| 解析结果 | `read_scene` / `{"scene": "神庙大厅"}`，`ok = true` |

工具调用这条比自由文本更严：**只要错一个 token，JSON 就坏了**，解析会直接失败。
所以"逐 token 一致 + 解析成功"同时成立，说明渲染、解码、解析三段都对了。

### 16.6 已知限制

- **不做非文本 content**：多模态数组内容会被当成空串（与官方模板对非字符串的
  处理一致）。本引擎只处理文本。
- **不做通用模板**：只实现 Qwen3 这一个模板。换模型族要加一个渲染函数，不是
  配一个模板 —— 这是刻意的（见 §6：不引入 Jinja）。
- **空 messages** 会渲染出生成提示而不是报错。官方 HF 是直接抛异常
  （"Cannot apply chat template to an empty conversation"），所以这条没有对拍
  参照，属于我们自己的约定。
- **解析器不校验 name 是否在工具集里** —— 那是宿主的护栏（对齐"护栏做在工具
  体里"），引擎只负责把名字和参数取出来。

### 16.7 下一步（M4）

把 `src/engine_stub.cc` 换成真实现：模型注册表 + 会话注册表 + ABI 边界异常翻译。
M3 已经把 M4 需要的东西都准备好了 —— `ParseMessages` / `ParseTools` /
`RenderChatml` / `ParseModelOutput` 就是 `rpg_engine_generate` 的主体，
剩下的主要是句柄管理、最长公共前缀匹配与 JSON 请求/响应的拼装。

注意 M4 解码时**不要**用 `skip_special`（理由见 §16.4）。

---

## 17. M4 落地记录（已完成）

### 17.1 交付物

| 文件 | 内容 |
| --- | --- |
| `src/engine.cc` | C ABI 的真实现（替换 `src/engine_stub.cc`）。M3 做好的 `ParseMessages` / `ParseTools` / `RenderChatml` / `ParseModelOutput` 直接成为 `rpg_engine_generate` 的主体 |
| `tools/check_abi.py` | 第 9 层：宿主视角（ctypes 走 dll）的端到端验收 |
| `tools/run_checks.sh` | 八层 → 九层；新增 `RPG_SKIP_ABI=1` |
| `CMakeLists.txt` | 源清单同步（`src/engine_stub.cc` → `src/engine.cc`） |

`include/rpg_engine.h` 与 `src/rpg_engine.def` 都不用改 —— M0 定的 10 个函数签名
在 M4 一个都没动，这正是"先把 ABI 钉死再实现"的价值：宿主侧的绑定（Unity C#
P/Invoke、Python ctypes）在 M0 就能写完并跑通调用，不必等内核。

### 17.2 三个落地决定

**① 句柄只增不复用。** 自增整数，销毁后永不回收。宿主拿旧句柄再调用一定失败，
而不是像复用的文件描述符那样悄悄打到另一个对象上。代价是极端情况下会耗尽
`int32`（那需要创建 20 亿个会话），已在分配处显式检查并报错。

**② 注册表用一把锁，但只在查找时持有。** 一次 `rpg_engine_generate` 在 CPU 上要跑
几十秒，握着锁跑完就等于"多会话"名存实亡。所以锁只保护句柄 → 对象的映射，
会话自身的状态按头文件的约定由宿主串行化。

**③ 所有"会失败的条件检查"必须排在"动 cache"之前。** 这一条是被一次跑红逼出来的
（见 §17.3）。

### 17.3 前缀复用：一次跑红换来的两条约束

第 9 层第一次跑的结果是：冷启动全过，**同一会话上的第二次调用返回 NULL**，
第三次报"内部状态不一致"。根因是两件事叠在一起：

**其一，`lcp == total` 时没有"下一个 token 的分布"。** 宿主原样重发同一份请求时，
整个 prompt 都是缓存前缀，`rows = total - lcp = 0` —— 而 `ForwardFull` 明确
`RPG_CHECK(seq > 0, "prompt 为空")`。就算它肯算，`logits.data() + (rows - 1) * vocab`
也已经越界读到数组前面去了。

修法是**回退一个位置**：`if (lcp == total) lcp = total - 1;`。重算一个位置就能拿到
那个位置的分布，代价约 0.1 秒。这里刻意**不去缓存上次的 logits 行** —— 多存一份
状态就多一处要同步的地方（工程上这类"顺手加个缓存"通常是后面某个难查 bug 的起点），
而收益只是省一次单 token 前向。

**其二，失败必须让会话回到干净状态。** 上面那次失败发生在 `kv->Reset()` +
`SetLen(lcp)` 之后、`cached_ids` 更新之前，于是会话停在了
`cached_ids.size() = 324` 而 `kv.len() = 302` 的状态里 —— 下一次调用被
`RPG_CHECK(cached == s->kv->len())` 当场拦下。所以：

- 条件检查（长度、容量）全部前移到动 cache 之前；
- 前向本身包在 `try/catch` 里，失败即 `kv->Reset()` + `cached_ids.clear()`，
  宁可下一次从零重来，也不让缓存带着说不清的长度继续用。

那条 `cached == s->kv->len()` 的不变量是值得留着的：它把"状态错乱"从"输出慢慢变味"
变成了"下一次调用当场报错"。

### 17.4 验证结果

第 9 层（`tools/check_abi.py`，**35/35 通过** —— 末尾会打印通过项数）。它不需要
torch，所以在前 8 层跑不起来的机器上照样能跑。覆盖：

| 组 | 内容 |
| --- | --- |
| 导出表 | 头文件声明 ↔ `.def` 清单 ↔ dll 实际导出三方一致（10 个函数）；全部可从 dll 解析到 |
| 错误通道 | 无效模型 / 无效会话句柄 → 失败返回 + `last_error` 非空 |
| 模型信息 | 28 层 / hidden 2048 / head_dim 128 / vocab 151936 / tied=True / 初始会话数 0 |
| 生成结构 | 用例产出 `tool_calls[0].name = read_scene`、参数 `{scene: 神庙大厅}`、`finish_reason = tool_calls`、`usage = {302, 22}` |
| 与工具链一致 | 同一份请求走"命令行工具链"与走 dll，`text` 与 `tool_calls` **完全相同** |
| 响应形状 | 响应**恰好四键**、每个 `tool_call` **恰好两键**；解析层非法分支上有日志且带原文（静态守卫） |
| 前缀复用 | 全前缀复用（重发同一请求）结果不变；部分前缀复用（续写一轮）与冷启动**完全相同** |
| 生命周期 | `session_reset` 后结果回到冷启动值；`destroy` / `unload_model` 之后旧句柄全部失效；句柄 0 无效 |

### 17.5 两条接口决定（已定案：都不加字段）

M4 落地时冒出来两个"要不要往响应里加字段"的问题。定案都是**不加字段**，把它们
变成日志或干脆不留 —— 响应是给游戏的，日志才是给排错的。

**① 不下发 `reasoning`（思考块内容）。** 响应保持 `text` / `tool_calls` /
`finish_reason` / `usage` 四键。理由是**防剧透**：响应最终会流到玩家侧 UI，而 DM
的思考里常常直接写着它打算怎么推进剧情；小助手更是本来就该与玩家同视角（见主项目
`ENGINE-ARCHITECTURE.md` 的"小助手可见范围不得超过玩家"）。想调 DM 的时候想看它
在想什么 —— 那是**调试需求，走日志**，不走响应。解析器内部仍然保留
`ParsedOutput.reasoning`（它得靠它把思考块从 `text` 里剥干净），只是不出 ABI。

**② 非法参数的工具调用不把原文塞进响应。** 那一条只给 `arguments: null`，模型的
原始输出（`ParsedToolCall.arguments_raw`）**记进日志**（`RPG_LOG_WARN`，默认级别
可见），带上工具名与原文，供"模型偶尔不按格式出牌"时定位。宿主从响应里拿到的永远
是干净的"这次调用不可用"，排错线索在日志里。

两条都加了防回归守卫（正是"删了不会让别的检查变红"的那类约定）：
`tools/check_abi.py` 断言响应**恰好四键**、每个 `tool_call` **恰好两键**；
`parser_logs_invalid_calls()` 静态确认解析层的非法分支上仍有 `RPG_LOG_WARN`
且带上了 `arguments_raw`。

### 17.6 已知限制

- **`temperature` / `top_p` 收下但不生效**：M2.5 之前一律按贪心跑。请求里给了这些
  字段不会报错，但也不会改变输出 —— 这一点在 README 的限制里写明了。
- **`max_seq` 不能超过 `max_pos`**：RoPE 表按 `context_length` 截断，缓存超出可用
  位置数没有意义。超了会被收窄并打一条 warn，不报错。

### 17.7 下一步（M2.5）

采样（温度 / top-k / top-p）的接口形状在 §12 #6 已经定案：**引擎只做确定性筛选，
抽样由宿主给一个 `[0,1)`**。所以 M2.5 要做的是：

- `GenerateOptions` 增加 `temperature` / `top_k` / `top_p`，实现"算分布 + 筛选"；
- ABI 上再加一个入口或一个可选字段，让宿主把它的随机数递进来（**形状待定，
  动之前先确认**）；
- 判据：`[0,1)` 取固定值时逐 token 可复现；与官方 `transformers` 的
  `temperature` + `top_p` + `top_k` 组合逐 token 对拍。

**宿主侧口径（用户 2026-10-11）**：玩家调创造力的控件应该是"**滑动变阻器**"式的
—— 连续、线性、拧到一端就是完全确定。这一条约束的是**接口的简单性**：宿主（Unity
UI / 游戏设置）不该被迫自己算概率、做逆变换或理解 top-k/top-p 的相互作用，它只该
把一个"旋钮位置"递给引擎。落到 ABI 上是加一个参数还是加一个入口，等确认后再动。

