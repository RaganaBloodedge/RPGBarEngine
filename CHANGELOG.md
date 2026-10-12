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

## [0.4.1] - 2026-10-12

### 修复

- **前缀复用的第二次 prefill 会越界写 `scores`，破坏堆（进程崩在分配器里）。**
  `ForwardScratch::Ensure()` 会把 `scores` 按**本段长度** `seq` 缩容，但 attention
  是按**绝对位置**索引它的（`AttentionFromCache` 里 `s <= pos0 + t`），长度得是
  `pos0 + seq`。正常 prefill（`pos0 = 0`）看不出问题；一旦宿主按"每轮重发完整
  messages"的方式续写（`pos0` = 缓存长度），多出来的 `pos0` 个 float 就写到
  缓冲区后面去了。

  症状极隐蔽，值得记下来：**单轮 prompt 完全正常**，第二轮复用前缀时进程**无声消失**
  （Windows 上是 `0xC0000374` 堆损坏，或 `0xC0000005` 访问违例），崩溃点在堆分配器
  里、与注意力看不出关系；同一份代码换个 prompt 长度可能又不崩（取决于堆布局），
  极易被误判成"偶发""并发问题"甚至"模型太大"。

  修法：`scores` 交由 `EnsureScores()` 独占管理（只增不减，`Ensure()` 不再碰它），
  并要求 `ForwardFull` 按 `pos0 + seq` 备足。`ForwardStep` 里那句
  `EnsureScores(pos + 1)` 随之去掉 —— 它本来就是为这个越界打的补丁，但被 `Ensure`
  的缩容抵消掉了，所以只保护了增量解码、没保护带前缀复用的 prefill。

- 回归测试：`tools/selftest.cc` 新增 `TestForwardScratch`，锁**缓冲区尺寸契约**
  （`Ensure` 不动 scores、`EnsureScores` 只增不减、前缀复用时按 `pos0 + seq` 覆盖）。
  不从数值上验 —— 越界写的是 `scores` 之外的堆，算出来的 logits 照样是对的，
  数值对拍抓不到这个 bug。

---

## [0.4.0] - 2026-10-11

**M5 完成**：prefill 从 7.5 tok/s 提到 **85–91 tok/s**（累计 11.7×），decode 保持 22.9 tok/s。

### 新增

- **多 token 阻塞乘内核**（`MatMulBlocked`，Q4_K / Q6_K / Q8_0 / F32）：一次权重读
  服务 **8 个 token**，把 prefill 的权重流量降到 1/8。这是 prefill 慢的真正原因 ——
  不是内核慢，是每个 token 都把整份权重重扫一遍。
- **`tools/prefill_prof`**（性能归因工具，不判定对错）：一次量出纯 FMA 上限（单线程与
  多线程）、顺序读带宽、端到端 prefill、matmul-only 占比、**线程扩展曲线**。M5 后半的
  两步归因完全靠它，过程记在 `docs/DESIGN.md` §19.7。

### 变更

- **`ScaleMinK4Inline` 就地内联**：原来调 `quant.cc` 的 `GetScaleMinK4Bytes` —— 那是
  **跨编译单元的真实函数调用**（无 LTO），Q4_K 每块调 8 次，单次扫权重约 18.7 亿次。
  热循环里没有"次数少所以无所谓"的函数。单线程内核 5.5 → **10.2 GMAC/s**。
- `MatMulBatch` 在 `n_tokens >= 8` 时优先走阻塞内核，逐 token 路径保留（单 token
  decode 仍走它，行为不变）。
- 自检 133 → **150** 项（+17：阻塞乘 vs 逐 token 一致、串行 vs 并行逐位相同、长度尾巴）。
- **删除 `quant.h/.cc` 里的 `GetScaleMinK4Bytes`**：内联后它成了死代码，留着就是一个
  现成的、看起来能安全复用的跨 TU 摘出物 —— 等于给后人在热循环里重挖同一个坑。
  6-bit 拆包现在只在 `matmul_avx2.cc` 里就地 `inline`。
- 版本号 `0.3.0` → `0.4.0`。

### 实测（Qwen3-1.7B-Q4_K_M，i7-14650HX / 16 线程）

| | 0.2.0（标量） | 0.3.0 | **0.4.0** |
| --- | --- | --- | --- |
| prefill 64 token | 8517 ms | 1711 ms | **726–747 ms（85–91 tok/s）** |
| matmul-only n=64 | 1682 ms | 1549 ms | **574–588 ms** |
| matmul 单线程 n=32 | 5.5 GMAC/s | 10.2 GMAC/s | **32.0 GMAC/s** |
| decode | 6.32 tok/s | 22.89 tok/s | 22.89 tok/s |

数值一致性未被提速破坏：第 6 层 argmax 全一致、第 7 层**逐 token 与 PyTorch 相同**、
第 9 层 C ABI 全过、十一层全绿。

### 未完成

- **`top_p` 仍是全量排序**（vocab 151936，每个采样 token 几毫秒，在 `temperature > 0`
  的热路径上）。改 partial sort 是独立小改。
- **prefill 85–91 未达 §9 的 100**：修完权重流量后瓶颈换成了**激活流量**（`MR=1` 每行
  重扫激活，线程扩展从 11× 掉到 5.3×）。提高 `MR` 要牺牲 `NR`，收益互相抵消，见 §19.7(b)。

---

## [未发布]

（暂无）

---

## [0.3.0] - 2026-10-11

性能（M5 的前一半）：AVX2 融合反量化点积内核。Qwen3-1.7B-Q4_K_M 上
**decode 6.3 → 22.9 tok/s（超设计目标 15）**、prefill 7.5 → 35.4 tok/s。
数值上与标量路径等价，端到端仍与 PyTorch 官方实现**逐 token 一致**。

### 新增

- **AVX2 融合反量化点积（M5）**：Q4_K / Q6_K / Q8_0 / F32 四个内核，在寄存器里
  把量化权重现解成 f32 并立刻 FMA，不再"解一行 → 落内存 → 再读回来"。
- **CPU 特性分派**：AVX2 + FMA 可用才走 SIMD，否则**自动退回标量**。库仍是
  零告警、全静态链接的单个 dll，在旧 CPU 上不会非法指令崩溃。
  环境变量 `RPGBAR_NO_SIMD=1` 可强制标量（同机对照、归因加速来源用）。
- **第 11 层验证**（`tools/check_dotrow.py`）：AVX2 内核 vs 标量参考 vs float64
  独立参考，判据 `max|Δ| / Σ|a·w| ≤ 1e-6`（实测 2.5e-8~4.9e-8）。
  导出探针 `tools/dotrow_check.exe` 从真实张量取权重行，不用造的假数据。
- `selftest` 从 124 条加到 **133 条**：新增的内核一致性断言**不需要权重**，
  改内核后 1 秒内就能发现位段取错。

### 变更

- `DotRowQuant` 由 `quant.cc` 迁到新增的 `matmul.h` / `matmul.cc`。
  `quant.cc` 只保留与 SIMD 无关的反量化 —— 它是对拍基准，必须保持纯标量。
- `quant.h` 新增 `kMaxBlockElems` 与 `GetScaleMinK4Bytes`（后者导出给 SIMD 内核
  复用它那份已经验过的 6-bit 拆包）。
- 构建：只有 `matmul_avx2.cc` 开 `-mavx2 -mfma`，其余编译单元保持基线指令集；
  编译器不支持时自动不加（内核退化为标量）。
- `build.sh` 改为"先链到 `*.new` 再 `mv -f` 就位"。原因是链接器没法**覆盖**
  已存在的产物时只会回一句 `ld returned 1 exit status`、把自己真正的诊断吞掉。

### 未完成（属 M5 的另一半）

- **prefill 仍慢在访存**：现在每个 token 都把整份权重重扫一遍，没做多 token 分块，
  35.4 tok/s 离 100 的目标仍差。**不是内核慢。**
- **top_p 仍用全量排序**（vocab 151936），每个采样 token 多花几毫秒。

---

## [0.2.0] - 2026-10-11

采样落地：引擎现在能按宿主给的 `temperature` 调节"创造性"，而随机性来自一个
counter-based 纯函数 —— 库里依然没有有状态的 RNG。同时定案了响应形状。

### 新增

- **采样（M2.5）**：`temperature` / `top_k` / `top_p` 的确定性筛选
  （`FilterToProbs`）与之后的逆变换。`generate` 请求里给 `"temperature": 0.8`
  即启用；`<= 0`（默认）仍走贪心，与之前逐位一致。
- **随机源**：宿主给一个 `seed`（请求里的 `seed` 字段，或 `session_json.seed`），
  引擎按 `u = H(seed, KV Cache 绝对位置)` 展开 —— 纯函数，同 seed + 同 prompt
  必得同一局；暖缓存（续写）与冷缓存（从头）逐位一致。**宿主不需要自己攒随机数。**
- **引擎侧旋钮兜底**：`top_p = 0.95`、`top_k` 关闭、温度钳在 `[0, 2]`（超限打警告）。
  宿主侧只暴露 `temperature` 一个旋钮，不必理解三者的相互作用。
- 第 10 层验收 `tools/check_sample.py`：108 组（3 温度 × 3 top_k × 3 top_p ×
  4 种 logits）的筛选结果与 HF 的 `Temperature`/`TopK`/`TopP` warpers
  **逐元素一致**；随机源验其代数性质（纯函数 / 均匀）。这一层**不需要权重**。

### 变更

- **响应形状定案：不加字段，改为记日志。** `rpg_engine_generate` 的响应**保持**
  `text` / `tool_calls` / `finish_reason` / `usage` 四键，两处曾经的待定项都落在
  "不加字段"上：
  - **思考块内容（`reasoning`）不下发** —— 响应最终会流到玩家侧 UI，而 DM 的思考
    里常常写着它打算怎么推进剧情，放进去等于漏剧本；小助手更是本来就该与玩家同
    视角。想看模型在想什么，走日志。
  - **非法参数的工具调用不给原文** —— 那一条仍给 `arguments: null`，模型原始输出
    记进日志（`RPG_LOG_WARN`，默认级别可见），带工具名与原文，供"模型偶尔不按
    格式出牌"时定位。
- `tools/check_abi.py` 新增三条守卫（响应恰好四键、每个 `tool_call` 恰好两键、
  解析层非法分支上仍有日志且带原文），并在末尾打印通过项数；第 9 层 32 → **35 项**。
- 第 1 层自检 106 → **124 项**（新增 16 条采样断言）。
- `docs/DESIGN.md` §12 #6 修正：原定案"宿主传一个 `[0,1)`"改为"宿主传 seed"，
  修正经过见 §18.2。

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
