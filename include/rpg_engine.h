/* rpg_engine.h —— RPGBar 内置推理引擎的 C ABI。
 *
 * 只暴露 C 接口，不暴露任何 C++ 类型，避免不同编译器 / STL 版本之间的 ABI 问题。
 * 宿主侧：Unity C#（P/Invoke）、Python（ctypes）都能直接绑定。
 *
 * 两级句柄：
 *   model   —— 一份加载好的权重（整个进程通常只有一个）
 *   session —— 模型上的一个独立推理上下文（独立 KV Cache 与消息历史）
 *
 * 「一个权重、多个隔离会话」是刻意的：主机端 DM 与小助手共用同一份 4B 权重，
 * 但两者的会话必须完全隔离，不得互相看到对方的上下文。
 *
 * 线程模型：模型可承载多个会话；单个会话同一时刻只允许一个 generate 调用
 * （宿主侧串行化）。GEMM 内部自行使用线程池。
 */
#ifndef RPGBAR_ENGINE_H
#define RPGBAR_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 句柄约定：>= 0 为有效，< 0 为失败。 */
#define RPG_ENGINE_OK 0

/* ---------------------------------------------------------------- 版本 */

/* 引擎版本字符串，如 "0.1.0"。 */
const char* rpg_engine_version(void);

/* 最近一次错误信息（线程局部，勿释放，下一次调用会被覆盖）。 */
const char* rpg_engine_last_error(void);

/* ---------------------------------------------------------------- 模型 */

/* 加载权重。model_path 为 GGUF 文件路径；config_json 可覆盖运行时参数
 * （线程数、上下文长度等），传 NULL 用默认值。
 * 返回模型句柄，失败返回负数。 */
int32_t rpg_engine_load_model(const char* model_path, const char* config_json);

/* 卸载模型，并销毁其下所有会话。model 无效时为空操作。 */
void rpg_engine_unload_model(int32_t model);

/* ---------------------------------------------------------------- 会话 */

/* 在模型上创建会话。session_json 可指定该会话的生成参数默认值，传 NULL 用默认值。
 * 返回会话句柄，失败返回负数。 */
int32_t rpg_engine_session_create(int32_t model, const char* session_json);

/* 销毁会话并释放其 KV Cache。 */
void rpg_engine_session_destroy(int32_t session);

/* 清空会话的 KV Cache 与消息历史，保留配置与会话句柄本身。 */
int32_t rpg_engine_session_reset(int32_t session);

/* ---------------------------------------------------------------- 生成 */

/* 执行一次生成。request_json / 返回值为 UTF-8 JSON。
 *
 * 请求：
 *   {
 *     "messages": [ {"role":"system","content":"..."}, {"role":"user","content":"..."} ],
 *     "tools":    [ {"type":"function","function":{...}} ],   // 可选；给了就允许工具调用
 *     "max_tokens": 256,
 *     "temperature": 0.7,
 *     "top_p": 0.9,
 *     "stop": ["<|im_end|>"]
 *   }
 *
 * 响应：
 *   {
 *     "text": "...",                       // 已剥离思考块与工具调用块的可读文本
 *     "tool_calls": [ {"name":"read_scene","arguments":{...}} ],   // 无调用则为空数组
 *     "finish_reason": "stop" | "length" | "tool_calls",
 *     "usage": {"prompt_tokens": 123, "completion_tokens": 45}
 *   }
 *
 * 返回新分配的字符串，调用方用 rpg_engine_free 释放；失败返回 NULL，
 * 错误详情走 rpg_engine_last_error()。
 *
 * 会话按轮次追加：调用方把完整的 messages 传进来即可，引擎负责与已有 KV Cache
 * 做最长公共前缀匹配以复用（或由调用方用 session_reset 显式重置）。 */
char* rpg_engine_generate(int32_t session, const char* request_json);

/* 释放由引擎分配的字符串（rpg_engine_generate 的返回值）。 */
void rpg_engine_free(char* buf);

/* ---------------------------------------------------------------- 自省 */

/* 返回模型元信息 JSON（层数、维度、上下文长度、词表大小等），用于宿主自检。
 * 调用方负责 rpg_engine_free。 */
char* rpg_engine_model_info(int32_t model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RPGBAR_ENGINE_H */
