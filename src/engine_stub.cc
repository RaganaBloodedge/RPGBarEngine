/* engine_stub.cc —— C ABI 的占位实现。
 *
 * 目的有两个：
 * 1) 让构建链先通（M6 的交叉编译问题提前暴露，不留到最后）；
 * 2) 让宿主侧（Unity C# / Python）现在就能把绑定写出来并跑通调用，不必等内核完成。
 *
 * 除 version / last_error 外，全部按"未实现"返回失败 —— 不是可用的引擎。
 */
#include "rpg_engine.h"

#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(const char* msg) { g_last_error = msg ? msg : ""; }

constexpr int32_t kNotImplemented = -2;

}  // namespace

extern "C" {

const char* rpg_engine_version(void) { return "0.1.0-stub"; }

const char* rpg_engine_last_error(void) { return g_last_error.c_str(); }

int32_t rpg_engine_load_model(const char* model_path, const char* config_json) {
  (void)model_path;
  (void)config_json;
  set_error("rpg_engine_load_model: not implemented");
  return kNotImplemented;
}

void rpg_engine_unload_model(int32_t model) { (void)model; }

int32_t rpg_engine_session_create(int32_t model, const char* session_json) {
  (void)model;
  (void)session_json;
  set_error("rpg_engine_session_create: not implemented");
  return kNotImplemented;
}

void rpg_engine_session_destroy(int32_t session) { (void)session; }

int32_t rpg_engine_session_reset(int32_t session) {
  (void)session;
  set_error("rpg_engine_session_reset: not implemented");
  return kNotImplemented;
}

char* rpg_engine_generate(int32_t session, const char* request_json) {
  (void)session;
  (void)request_json;
  set_error("rpg_engine_generate: not implemented");
  return nullptr;
}

void rpg_engine_free(char* buf) { delete[] buf; }

char* rpg_engine_model_info(int32_t model) {
  (void)model;
  set_error("rpg_engine_model_info: not implemented");
  return nullptr;
}

}  // extern "C"
