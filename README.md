# RPGBarEngine

RPGBar 的内置推理引擎。目标是把"玩家要自己配模型服务"变成"装完游戏就能选到一个模型"——
引擎编进游戏本体，权重随包分发，进程内调用，零外部依赖、零 HTTP 往返。

> 状态：**M0 完成**（GGUF 解析 / 15 种量化类型反量化 / Qwen BBPE 分词器，全部通过独立验证）。
> 本地仓库已初始化，**暂不设 remote、不上传 GitHub**（按项目决定）。
> 进度与设计依据见 `docs/DESIGN.md`。

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
| `kv_cache.h`（KV Cache 设计） | 可参考 | 布局要按 Qwen3 的 GQA 重算 |
| `sbs_reader.cc`（.sbs 解析） | 不可用 | 已换成 GGUF 解析 |
| `model.cc`（Gemma 前向） | 不可用 | Qwen3 是另一套算子 |
| `tools/spm.py`（SentencePiece） | 不可用 | Qwen 用 BBPE，另写 |

## 目录

```
RPGBarEngine/
├── include/rpg_engine.h    C ABI（Unity C# P/Invoke 与 Python ctypes 都能绑）
├── src/                    引擎实现（见 DESIGN §13.1 的模块清单）
├── tools/                  检查工具与对拍脚本
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

产物：`build/rpg_engine.dll`、`build/dump_gguf.exe`、`build/tok_check.exe`、`build/selftest.exe`。

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

一次跑完四层检查，缺哪一层会明确说明而不是静默跳过：

| 层 | 手段 | 当前结果 |
| --- | --- | --- |
| 进程内自检 | `selftest.exe` | 69/69 |
| 量化块尺寸 | `check_quant_layout.py` 用 GGUF 写入器的 offset 反推 | 全部一致，张量无缝铺满 |
| 分词器 | `tok_ref.py` 对 HF `tokenizers` | 3031/3031 逐 id 一致 |
| 量化语义 | `xcheck_quant.py`：Q8_0 与 Q4_K_M 同一张量 | 相关系数 0.9966–0.9998，F32 逐位相同 |

单跑某一件：

```bash
./build/dump_gguf.exe models/Qwen3-1.7B-Q8_0.gguf --tensors --meta
./build/dump_gguf.exe models/Qwen3-1.7B-Q8_0.gguf --sample blk.0.ffn_gate.weight 16
python tools/tok_ref.py models/Qwen3-1.7B-Q8_0.gguf models/qwen3-1.7b-tokenizer.json --fuzz 3000
```

## 当前限制

- **不做 NFC 归一化**：跟随 llama.cpp 的行为（Qwen3 的 tokenizer.json 声明了 NFC，
  但 llama.cpp 不归一化）。输入已是 NFC 时与 HF 完全一致。详见 `docs/DESIGN.md` §13.4。
- **C ABI 还是占位实现**：除 `rpg_engine_version` 外都返回"未实现"，M4 落地。
- 只支持 Qwen3 架构；配置全部从 GGUF 读，扩族是"加一个前向 + 一套模板"，不是重写。

里程碑与验收判据见 `docs/DESIGN.md` §10，M0 的完整落地记录见 §13。
