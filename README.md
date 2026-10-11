# RPGBarEngine

RPGBar 的内置推理引擎。目标是把"玩家要自己配模型服务"变成"装完游戏就能选到一个模型"——
引擎编进游戏本体，权重随包分发，进程内调用，零外部依赖、零 HTTP 往返。

> 状态：**M3 完成**（ChatML 渲染与官方模板逐字节一致，工具调用端到端打通）。
> 已完成：GGUF 解析 / 15 种 ggml 量化类型反量化 / Qwen BBPE 分词器 /
> 全序列前向 / KV Cache / 贪心生成 / ChatML 模板 / tool_call 解析。
> 本地仓库已初始化，**暂不设 remote、不上传 GitHub**（按项目决定）。
> 进度与设计依据见 `docs/DESIGN.md`：M0 见 §13，M1 见 §14，M2 见 §15，M3 见 §16。
> **引擎不产随机性**（库里没有 RNG）—— 见下面「关于随机性」。

## 这是什么

一个 CPU 上的小型 LLM 推理引擎，服务于 RPGBar 的两个 Agent：

| 规模 | 模型 | 用在哪 | 角色 |
| --- | --- | --- | --- |
| 小 | Qwen3-1.7B | 玩家端：本地行动建议 | 小助手 Agent |
| 大 | Qwen3-4B | 主机端：DM + 房间级任务 | DM Agent + 小助手 Agent |

**每端只加载一个权重**：主机端加载大的（要跑 DM），玩家端加载小的（只跑本地建议）。
主机端的 DM 与小助手共用同一份权重，但**会话完全隔离**（两级句柄：`model` 管权重、
`session` 管独立 KV Cache 与消息历史）。

## 为什么换族（不是 Gemma）

原路线是复用 `TinyGemmaCpp`（Gemma 2 架构）。但 **Gemma 2 没有原生工具调用模板**——
`chat_with_tools` 在本地引擎上必然失败，DM 会退化到"只能输出文本、改不了状态机"。

DM 的全部价值就在于多轮工具循环（`read_scene` → `roll_check` → `move_to` → `set_flag`），
所以**工具调用是硬需求**，必须换成原生支持的模型族。详见 `docs/DESIGN.md` §2。

## 与 TinyGemmaCpp 的关系

**不复用代码，只复用工程经验。** 两者架构不同（Gemma 2 vs Qwen3），权重格式不同
（`.sbs` vs GGUF），分词器不同（SentencePiece vs BBPE）：

| 模块 | 能否复用 | 说明 |
| --- | --- | --- |
| `matmul.cc`（GEMM + AVX2/FMA 优化） | M5 移植 | 与模型架构无关 |
| `threads.cc`（物理核探测） | 已移植 | 但并行后端换成自研线程池（见 DESIGN §13.3） |
| `dtypes.h`（bf16/fp16 类型） | 已移植并扩展 | 补了 f16，去掉 `.sbs` 专有的 sfp |
| `kv_cache`（KV Cache 设计） | 参考了思路 | 布局要按 Qwen3 的 GQA 重算，是另写的（`src/kv_cache.*`） |
| `sbs_reader.cc`（.sbs 解析） | 不可用 | 已换成 GGUF 解析 |
| `model.cc`（Gemma 前向） | 不可用 | Qwen3 是另一套算子 |
| `tools/spm.py`（SentencePiece） | 不可用 | Qwen 用 BBPE，另写 |

## 目录

```
RPGBarEngine/
├── include/rpg_engine.h    C ABI（Unity C# P/Invoke 与 Python ctypes 都能绑）
├── src/                    引擎实现（见 DESIGN §13.1 / §14.1 的模块清单）
├── tools/                  检查工具与对拍脚本
├── tests/prompts/          对拍用的提示词
├── tests/chatml/           ChatML 对拍用例（带工具定义的场景）
├── scripts/fetch_model.py  权重与对拍参考物下载（断点续传）
├── models/                 权重与参考物（gitignore，见下）
└── docs/DESIGN.md          设计与选型依据
```

## 怎么开始

### 1. 构建

有 cmake 的环境：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

本机（MSYS2，没有 cmake）：

```bash
./build.sh
```

产物：`build/rpg_engine.dll`、`build/dump_gguf.exe`、`build/tok_check.exe`、
`build/selftest.exe`、`build/forward_check.exe`、`build/gen_check.exe`、
`build/chatml_check.exe`、`build/parse_check.exe`。

两个构建坑（改构建脚本时别丢）：

- **必须全静态链接**（`-static`）。否则 dll 依赖 MSYS2 的 `libwinpthread-1.dll`，
  宿主加载时报 `Could not find module ... or one of its dependencies`。
  本引擎用了 `std::thread`，所以 libwinpthread 绕不开，只能静态吃进去。
- **Windows 导出必须走 `src/rpg_engine.def`**。不写 `.def` 时 MinGW ld 会导出全部
  符号（实测 200+ 条，含内部 C++ 修饰名），宿主可能误解析到内部符号。

### 2. 取权重

权重不进版本库（`models/` 已 gitignore）。用下载脚本：

```bash
python scripts/fetch_model.py --all-small     # Q8_0（对拍基准）+ Q4_K_M（出货档）
python scripts/fetch_model.py --list          # 看所有 preset
```

脚本默认走 `hf-mirror.com`（国内直连 huggingface.co 不通），带断点续传。

### 3. 验证

```bash
bash tools/run_checks.sh
```

一次跑完八层检查，缺哪一层会明确说明而不是静默跳过：

| 层 | 手段 | 当前结果 |
| --- | --- | --- |
| 1 进程内自检 | `selftest.exe` | 106/106 |
| 2 量化块尺寸 | `check_quant_layout.py` 用 GGUF 写入器的 offset 反推 | 全部一致，张量无缝铺满 |
| 3 量化语义（逐位） | `xcheck_dequant.py` vs `gguf.quants`（llama.cpp 官方算法） | **逐位相同**（含 3.1 亿元素的整张词表） |
| 4 分词器 | `tok_ref.py` 对 HF `tokenizers` | 3031/3031 逐 id 一致 |
| 5 跨量化档位 | `xcheck_quant.py`：Q8_0 与 Q4_K_M 同一张量 | 相关系数 0.9966–0.9998，F32 逐位相同 |
| 6 前向（M1） | `check_forward.py` vs PyTorch 官方 `Qwen3ForCausalLM` | 3 组合 × 29 项全绿，argmax 全一致 |
| 7 生成（M2） | `check_generate.py` vs PyTorch 官方实现 | 3 组合 **逐 token 一致**，cache 自检逐位相同 |
| 8 ChatML 渲染（M3） | `check_chatml.py` vs HF `apply_chat_template` 跑官方模板原文 | 15/15 用例逐**字节**一致 |

第 6 层之后的都需要 `torch` / `transformers` / `gguf`；第 8 层另需 `jinja2`：

```bash
pip install torch transformers gguf safetensors
```

单跑某一件：

```bash
./build/dump_gguf.exe models/Qwen3-1.7B-Q8_0.gguf --tensors --meta
./build/dump_gguf.exe models/Qwen3-1.7B-Q8_0.gguf --dump-npy blk.0.ffn_gate.weight /tmp/w.npy
./build/forward_check.exe models/Qwen3-1.7B-Q8_0.gguf --prompt "The capital of France is" --top 5
./build/forward_check.exe models/Qwen3-1.7B-Q8_0.gguf --prompt-file tests/prompts/zh_chatml.txt --out runs/zh
./build/gen_check.exe models/Qwen3-1.7B-Q8_0.gguf --prompt "The capital of France is" --max-tokens 32
python tools/check_forward.py                       # M1 全部组合
python tools/check_generate.py                      # M2 全部组合
python tools/check_chatml.py                        # M3 第 8 层（逐字节）
python tools/check_toolcall.py                      # M3 端到端（工具调用）
./build/chatml_check.exe --case tests/chatml/dm_toolcall.case.json   # 渲染一份 ChatML
./build/parse_check.exe --text-file runs/xxx.raw.txt                 # 解析一段输出
python tools/check_noise_floor.py models/Qwen3-1.7B-Q8_0.gguf --layers 12 --tokens-from runs/zh.tokens.txt
```

> Windows 下**中文提示词不要走命令行参数**：`cmd` / Git Bash 会按控制台代码页转码，
> 传进去已经是乱码字节了。用 `--prompt-file` 从文件读。

### 关于 M1 的误差判据

设计里原本写的是"逐张量误差 < 1e-4"。实测逐层残差流的 `max|Δ|` 是 `0.0117`，
超限 100 倍 —— 但 top-5 token 与官方实现完全一致。原因是 **Qwen3 的残差流里有一个
量级 ~1.2e4 的离群维度**，在那个量级上 float32 的 1 ULP ≈ 1e-3，
"绝对 1e-4" 比一个 ULP 还小 10 倍，不可能满足。

判据已改为**逐元素相对误差 p99 ≤ 2e-4**（这个阈值的来源是把 float32 的噪声底
实测出来，见 `docs/DESIGN.md` §14.3）。当前最差 **1.47e-4**，argmax 全位置一致。

### 关于 ChatML 与工具调用

引擎**只解析、不执行**。工具的权限校验、状态写入全在宿主 —— 对齐项目里
"护栏做在工具体里，不做在提示词里"那条纪律。

```
宿主传 messages + tools（JSON Schema）
   │
   ├─ RenderChatml   按 Qwen3 模板渲染（工具定义进 system 消息）
   ▼ 生成
   ├─ ParseModelOutput  取出 <tool_call>{...}</tool_call>，剩下的就是"该说的话"
   ▼
{"text": "...", "reasoning": "...", "tool_calls": [{"name":..., "arguments":{...}}]}
```

**渲染是逐字节对着官方模板做的。** 参考侧直接取官方 `tokenizer_config.json` 里的
`chat_template` 原文交给 HF 渲染，所以比的是模板本身。两处凭记忆一定会写错的细节：

- 空思考块是 **`<think>\n\n</think>\n\n`**，特殊 token 就叫 `<think>` / `</think>`，
  **没有前导空格**（不是 ` thinking`）。
- assistant 消息**只有在"最后一次真实用户提问之后"**才渲染思考块，判断依据是官方
  模板里的 `last_query_index`。

⚠️ **给解析器喂的必须是 `skip_special = false` 的解码文本。** `<tool_call>` /
`</tool_call>` 自己就是特殊 token（151657/151658），开了 skip_special 会把结构标记
一起删掉，解析器只能看到裸 JSON、还会把它当旁白念给玩家听。控制标记由解析器自己清。
这条是 M3 端到端第一次跑红时踩出来的，详见 `docs/DESIGN.md` §16.4。

### 关于随机性：引擎里没有 RNG

**库里不含任何随机数生成器，也不该有。** 理由不是洁癖，是三条实际的：

1. **对齐 RPGBar 的"骰子只有一个来源"**。项目里已经有这条纪律：模型与引擎只产出
   "权重或分布"，从不自己掷骰（骰点走服务端 `dice.roll_check`，NPC 反应落点走
   `dice.weighted_pick`）。引擎若自建一套 RNG，就是在同一条纪律上开口子。
2. **可复现**。随机流集中在宿主侧，才能做到"同一个房间种子 + 同一串玩家输入 →
   一模一样的剧情"。多会话（DM 与小助手共用一份权重）若共享一条 RNG，谁先谁后
   还会互相扰动随机流。
3. **引擎是纯函数**。同一输入必得同一输出 —— 这对本项目的对拍验证是前提
   （第 6、7 层的判据都建立在"确定性"上）。

所以分工是：

| 步骤 | 谁做 | 要随机数吗 |
| --- | --- | --- |
| 算 logits | 引擎（前向） | 不要 |
| 温度 / top-k / top-p 筛选 | 引擎（纯函数，**M2.5 实现**） | 不要 |
| 贪心取最大 | 引擎（M2 已实现） | 不要 |
| 按概率抽一个 | **宿主** | **要一个 `[0,1)`** |

宿主侧那个"创造力旋钮"就是 **temperature**：调到 0 走贪心（稳定，适合工具调用与
JSON 输出），调高则分布变平、更容易选到低概率的表达（更适合叙事）。DM 与小助手
可以各自设一个。设计原委见 `docs/DESIGN.md` §12 #6。

## 当前限制

- **只能贪心**：温度 / top-p / top-k 采样还没有代码（M2.5）。按上面的分工，即使做，
  引擎也只出确定性分布，抽样由宿主完成。
- **decode 与 prefill 都慢**：Q8_0 上 prefill 64 token 约 6.6 s、decode 约 9 tok/s，
  离目标 100 tok/s 差一个数量级。这是纯标量反量化点积的必然结果（读每个权重元素都要
  现反量化一次），M5 用 SIMD + 融合反量化点积解决。
- **只实现 Qwen3 一个模板**：不做通用模板引擎（不引 Jinja），换模型族要加一个
  渲染函数。这是刻意的取舍，见 `docs/DESIGN.md` §6。
- **不做 NFC 归一化**：跟随 llama.cpp 的行为（Qwen3 的 tokenizer.json 声明了 NFC，
  但 llama.cpp 不归一化）。输入已是 NFC 时与 HF 完全一致。详见 `docs/DESIGN.md` §13.4。
- **C ABI 还是占位实现**：除 `rpg_engine_version` 外都返回"未实现"，M4 落地。
- 只支持 Qwen3 架构；配置全部从 GGUF 读，扩族是"加一个前向 + 一套模板"，不是重写。

里程碑与验收判据见 `docs/DESIGN.md` §10；M0 记录见 §13，M1 见 §14，M2 见 §15，
M3 见 §16。
