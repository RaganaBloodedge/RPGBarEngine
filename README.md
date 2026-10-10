# RPGBarEngine

RPGBar 的内置推理引擎。目标是把"玩家要自己配模型服务"变成"装完游戏就能选到一个模型"——
引擎编进游戏本体，权重随包分发，进程内调用，零外部依赖、零 HTTP 往返。

> 状态：**设计阶段**。目前只有设计与接口（C ABI），尚无实现。
> 本地仓库已初始化，**暂不设 remote、不上传 GitHub**（按项目决定）。

## 这是什么

一个 CPU 上的小型 LLM 推理引擎，服务于 RPGBar 的两个 Agent：

| 规模 | 模型 | 用在哪 | 角色 |
| --- | --- | --- | --- |
| 小 | Qwen3-1.7B | 玩家端：本地行动建议 | 小助手 Agent |
| 大 | Qwen3-4B | 主机端：DM + 房间级任务 | DM Agent + 小助手 Agent |

**每端只加载一个权重**：主机端加载大的（要跑 DM），玩家端加载小的（只跑本地建议）。

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
| `matmul.cc`（GEMM + AVX2/FMA 优化） | 可移植 | 与模型架构无关 |
| `threads.cc`（线程池） | 可移植 | 同上 |
| `dtypes.h`（bf16/fp16 类型） | 可移植 | 同上 |
| `kv_cache.h`（KV Cache 设计） | 可参考 | 布局要按 Qwen3 重算 |
| `sbs_reader.cc`（.sbs 解析） | 不可用 | 换成 GGUF |
| `model.cc`（Gemma 前向） | 不可用 | Qwen3 是另一套算子 |
| `tools/spm.py`（SentencePiece） | 不可用 | Qwen 用 BBPE |

## 目录

```
RPGBarEngine/
├── include/rpg_engine.h    C ABI（宿主：Unity C# P/Invoke 与 Python ctypes 都能绑）
├── docs/DESIGN.md          设计与选型依据
├── src/                    实现（待建）
├── tests/                  对拍与微基准（待建）
└── tools/                  GGUF 检查、tokenizer 探针等脚本（待建）
```

## 怎么开始

有 cmake 的环境：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

本机（MSYS2，没有 cmake）：

```bash
./build.sh
```

骨架阶段的产物是 `build/rpg_engine.dll`（占位实现，除 `rpg_engine_version` 外都返回"未实现"）。
已实测：dll 能被 Python ctypes 加载，10 个 C 函数按原始名导出，调用链通。

**必须全静态链接**（`build.sh` 里的 `-static`）——否则 dll 会依赖 MSYS2 的 `libwinpthread-1.dll`，
宿主（Unity / Python）加载时报 `Could not find module ... or one of its dependencies`。

里程碑见 `docs/DESIGN.md` §10。
