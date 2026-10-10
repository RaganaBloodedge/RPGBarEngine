# RPGBarEngine 设计

> 状态：**设计稿 v1**（选型待确认）
> 关联：`RPGBar-Unity/docs/ENGINE-ARCHITECTURE.md`（Agent 与 Provider 层设计）

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

> 本文所有超参（层数 / hidden / 头数 / rope_theta）在**实现前必须用官方 `config.json` 与权重头核对**，
> 下表为工作假设：
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
        ├── Session          KV Cache + 消息历史 + 采样器状态
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
| **M1** | Qwen3 前向（无 KV Cache，标量） | 与 PyTorch 参考实现对拍，逐 tensor 误差 < 1e-4 |
| **M2** | KV Cache + 采样（贪心 / top-p） | 贪心解码 N 个 token 与官方逐 token 一致 |
| **M3** | ChatML 模板 + tool_call 解析 | 给定工具定义，能稳定产出结构化 `tool_calls` |
| **M4** | C ABI + 多会话 | 同一模型两个会话互不干扰（对拍验证） |
| **M5** | SIMD / 多线程 GEMM（移植 TinyGemmaCpp） | 达到 §9 性能目标 |
| **M6** | 交叉编译出 `rpg_engine.dll`，Unity P/Invoke 打通 | Unity 里调用成功生成一段合法 JSON |

---

## 11. 验证方法

沿用 TinyGemmaCpp 已验证的三层对拍（每层都留可复现脚本）：

| 层 | 对拍对象 | 判据 |
| --- | --- | --- |
| 解码 | 量化反解后的权重 | 逐**位模式**一致（不是"误差很小"） |
| 分词 | 官方 tokenizer | 逐 **id** 一致 |
| 端到端 | 官方实现（llama.cpp）贪心解码 | 逐 **token** 一致 |

**为什么必须对位对位**：量化解码有损，一旦解码有偏差，后面的"误差很小"可能是错的巧合。
只有先钉死"同一份权重解出来同一个数"，上层的数值差异才可归因。

---

## 12. 待确认

| # | 问题 | 建议 |
| --- | --- | --- |
| 1 | **模型族**：Qwen3（1.7B / 4B）还是别的 | **Qwen3** —— 档位最贴合、Apache-2.0、一套架构覆盖两档、原生工具调用且可控思考模式 |
| 2 | 量化档位：Q4_K_M 起步，是否同时出 Q5/Q8 | **先只做 Q4_K_M**，打通后再加档 |
| 3 | 上下文长度上限：2048 / 4096 / 8192 | **4096** 起（DM 要装剧本片段 + 历史），按显存/内存实测调 |
