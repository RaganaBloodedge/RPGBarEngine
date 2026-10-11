/* engine.cc —— C ABI 的实现（M4）。
 *
 * 这里是引擎与宿主的唯一交界面。三条贯穿全文件的约定：
 *
 * 1. **异常绝不穿过 extern "C"**。引擎内部用异常传错误（见 src/common.h），
 *    但宿主是 Unity IL2CPP 与 CPython —— 让 C++ 异常穿出 C ABI 到那边是未定义
 *    行为。每个入口都包 try/catch，翻成 "返回负数 / NULL + rpg_engine_last_error()"。
 *
 * 2. **句柄只增不复用**。句柄是自增整数，销毁后永不回收 —— 于是宿主拿旧句柄
 *    再调用时一定失败（而不是像复用的 fd 那样打到另一个对象上，那种错最难查）。
 *
 * 3. **本层不做业务判断**。工具允不允许、参数合不合法、状态怎么改，全在宿主
 *    （对齐项目里"护栏做在工具体里"那条纪律）。这里只做：解析请求 → 渲染 →
 *    分词 → 前向/生成 → 解析输出 → 拼响应。
 *
 * 解码**不加 skip_special**：`<tool_call>` / `</tool_call>` / `<think>` 自己就是
 * 特殊 token，跳过它们会把结构标记一起删掉。理由见 docs/DESIGN.md §16.4。
 */

#include "rpg_engine.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "chat_template.h"
#include "common.h"
#include "json.h"
#include "kv_cache.h"
#include "model.h"
#include "qwen3.h"
#include "tool_parser.h"

namespace {

// 引擎版本。与 CMakeLists.txt 的 project(VERSION) 保持一致。
constexpr const char* kEngineVersion = "0.4.0";

// 请求/会话配置里没给 max_tokens 时的默认值。
constexpr int32_t kDefaultMaxTokens = 256;
// KV Cache 里给停止标记与端点检查留的余量。
constexpr int64_t kSeqMargin = 4;

// 采样的引擎侧兜底。宿主只需要调 temperature —— top_k / top_p 由引擎钉住，
// 宿主不必理解两者怎么互相作用（DESIGN §12 #6 的"滑动变阻器"口径）。
constexpr int32_t kDefaultTopK = 0;     // 0 = 关闭 top_k
constexpr float kDefaultTopP = 0.95f;   // 核采样：挡住长尾的胡言乱语
// 温度上限。超过 2.0 之后输出基本读不通，钳住并打警告（是护栏，不是限制）。
constexpr float kMaxTemperature = 2.0f;

// ---------------------------------------------------------------- 注册表

struct Session {
  int32_t model = -1;

  // KV Cache 与它的配套缓冲。cache 是一份会话最大的内存占用（1.7B 在 4096
  // 长度下约 940 MB），所以它跟着会话走、会话销毁即释放。
  std::unique_ptr<rpg::KvCache> kv;
  rpg::ForwardScratch scratch;

  // **cache 里当前实际存着的 token 序列**。generate 靠它做最长公共前缀匹配：
  // 只有长度与内容都逐一对上才能复用前缀。这里和 kv->len() 必须始终相等，
  // 不一致意味着"复用了错误的 K/V"—— 那种错不崩，只让输出悄悄变味。
  std::vector<int32_t> cached_ids;

  // 会话级默认生成参数（session_json 可覆盖，请求里的同名字段优先级更高）。
  int32_t max_tokens = kDefaultMaxTokens;
  std::vector<std::string> stop;

  // 会话级默认采样参数。temperature <= 0 即贪心 —— 这是默认值，保证"不传
  // temperature 的老请求"行为不变（M4 之前一律贪心）。
  float temperature = 0.0f;
  uint64_t seed = 0;
};

struct ModelEntry {
  std::unique_ptr<rpg::Model> model;
  int32_t max_pos = 0;          // RoPE 表覆盖的位置数（已被 context_length 截过）
  int32_t default_max_seq = 0;  // 新建会话时的 KV Cache 容量
  std::vector<int32_t> sessions;  // 便于 unload 时级联销毁
};

std::mutex g_mutex;
std::unordered_map<int32_t, ModelEntry> g_models;
std::unordered_map<int32_t, std::unique_ptr<Session>> g_sessions;
int32_t g_next_model = 1;
int32_t g_next_session = 1;

// ------- 下面两个 *Locked 版本要求调用方已持 g_mutex（g_mutex 不是递归锁）
ModelEntry* ModelLocked(int32_t h) {
  const auto it = g_models.find(h);
  RPG_CHECK(it != g_models.end(), "无效的模型句柄 %d（可能已被 unload）", h);
  return &it->second;
}

Session* SessionLocked(int32_t h) {
  const auto it = g_sessions.find(h);
  RPG_CHECK(it != g_sessions.end(), "无效的会话句柄 %d（可能已被 destroy）", h);
  return it->second.get();
}

ModelEntry* RequireModel(int32_t h) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return ModelLocked(h);
}

Session* RequireSession(int32_t h) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return SessionLocked(h);
}

// ---------------------------------------------------------------- 小工具

// 引擎内部 new[] / delete[] 成对出现（rpg_engine_free 也在本文件），跨 dll
// 边界传出去的字符串不会落到另一个 CRT 的堆上。
char* DupForAbi(const std::string& s) {
  char* buf = new char[s.size() + 1];
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  return buf;
}

std::string JsonToAbiString(const rpg::Json& j) { return j.Dump(/*pretty=*/false); }

// 把一份配置 JSON（可为空）解析成对象。空串/空指针 = 全用默认值。
rpg::Json ParseConfigOrEmpty(const char* text, const char* what) {
  if (text == nullptr || *text == '\0') return rpg::Json::Object();
  std::string err;
  rpg::Json j = rpg::Json::Parse(text, &err);
  RPG_CHECK(err.empty(), "%s 不是合法 JSON：%s", what, err.c_str());
  RPG_CHECK(j.IsObject(), "%s 必须是 JSON 对象", what);
  return j;
}

// 停止 token 的来源有三处：模型自带 eos、`<|im_end|>`、以及请求/会话里给的
// 停止串。后两者里的字符串必须先变成 token id —— 引擎按 token 粒度停，
// 不做"回看最近 N 个 token 的文本匹配"，那是宿主的事。
void AppendStopString(const rpg::Tokenizer& tok, const std::string& s,
                      std::vector<int32_t>* out) {
  if (s.empty()) return;
  const int32_t direct = tok.IdOf(s);
  if (direct >= 0) {
    out->push_back(direct);
    return;
  }
  const std::vector<int32_t> ids = tok.Encode(s, /*parse_special=*/false);
  if (ids.size() == 1) {
    out->push_back(ids[0]);
    return;
  }
  RPG_LOG_WARN("停止串 \"%s\" 不是单个 token（编成 " RPG_SZ
               " 个），已忽略 —— 多 token 的停止串需要文本级回看，属宿主职责",
               s.c_str(), RPG_SZ_ARG(ids.size()));
}

void AppendStopStrings(const rpg::Tokenizer& tok, const rpg::Json& arr,
                       std::vector<int32_t>* out) {
  if (!arr.IsArray()) return;
  for (const rpg::Json& e : arr.elements()) {
    if (e.IsString()) AppendStopString(tok, e.AsString(), out);
  }
}

bool ContainsId(const std::vector<int32_t>& v, int32_t id) {
  return std::find(v.begin(), v.end(), id) != v.end();
}

// ---------------------------------------------------------------- 异常翻译

// 把 C++ 异常翻成 ABI 的失败返回（见文件头第 1 条）。
#define RPG_ABI_CATCH(ret)                    \
  catch (const std::exception& e) {           \
    ::rpg::SetLastError(e.what());            \
    return ret;                               \
  }                                           \
  catch (...) {                               \
    ::rpg::SetLastError("unknown exception"); \
    return ret;                               \
  }

#define RPG_ABI_CATCH_VOID()                  \
  catch (const std::exception& e) {           \
    ::rpg::SetLastError(e.what());            \
  }                                           \
  catch (...) {                               \
    ::rpg::SetLastError("unknown exception"); \
  }

}  // namespace

extern "C" {

const char* rpg_engine_version(void) { return kEngineVersion; }

const char* rpg_engine_last_error(void) { return rpg::LastError(); }

int32_t rpg_engine_load_model(const char* model_path, const char* config_json) {
  try {
    rpg::SetLastError("");
    RPG_CHECK(model_path != nullptr && *model_path != '\0', "model_path 为空");

    const rpg::Json cfg = ParseConfigOrEmpty(config_json, "config_json");
    const int threads = cfg.Get("threads").AsInt32(0);
    int32_t max_pos = cfg.Get("max_pos").AsInt32(4096);
    int32_t max_seq = cfg.Get("max_seq").AsInt32(0);
    RPG_CHECK(max_pos > 0, "config_json.max_pos 必须为正（收到 %d）", max_pos);

    auto entry = std::make_unique<ModelEntry>();
    entry->model =
        std::make_unique<rpg::Model>(rpg::Model::Load(model_path, threads, max_pos));

    const rpg::ModelConfig& c = entry->model->config();
    // RoPE 表按 context_length 截断（见 Model::Load），所以缓存容量也不能超过它。
    int32_t eff_max_pos = max_pos;
    if (c.context_length > 0 && c.context_length < eff_max_pos) {
      eff_max_pos = c.context_length;
    }
    if (max_seq <= 0) max_seq = eff_max_pos;
    if (max_seq > eff_max_pos) {
      RPG_LOG_WARN("config_json.max_seq=%d 超过可用位置数 %d，已收到 %d",
                   max_seq, eff_max_pos, eff_max_pos);
      max_seq = eff_max_pos;
    }
    entry->max_pos = eff_max_pos;
    entry->default_max_seq = max_seq;

    std::lock_guard<std::mutex> lk(g_mutex);
    RPG_CHECK(g_next_model > 0, "模型句柄已耗尽（进程内已创建 %d 个）",
              g_next_model - 1);
    const int32_t h = g_next_model++;
    g_models.emplace(h, std::move(*entry));

    RPG_LOG_INFO("已加载模型 #%d：%s（%d 层 / hidden %d / %d 头 / %d KV 头 / "
                 "词表 %d / 位置 %d / 会话默认容量 %d）",
                 h, c.name.empty() ? c.arch.c_str() : c.name.c_str(), c.n_layers,
                 c.hidden, c.n_heads, c.n_kv_heads, c.vocab, eff_max_pos, max_seq);
    return h;
  }
  RPG_ABI_CATCH(-1)
}

void rpg_engine_unload_model(int32_t model) {
  try {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto it = g_models.find(model);
    if (it == g_models.end()) return;  // 无效句柄是空操作（见头文件约定）
    for (const int32_t s : it->second.sessions) g_sessions.erase(s);
    const int32_t n = static_cast<int32_t>(it->second.sessions.size());
    g_models.erase(it);
    RPG_LOG_INFO("已卸载模型 #%d（连带销毁 %d 个会话）", model, n);
  }
  RPG_ABI_CATCH_VOID()
}

int32_t rpg_engine_session_create(int32_t model, const char* session_json) {
  try {
    rpg::SetLastError("");
    const rpg::Json cfg = ParseConfigOrEmpty(session_json, "session_json");

    std::lock_guard<std::mutex> lk(g_mutex);
    ModelEntry* m = ModelLocked(model);

    int32_t max_seq = cfg.Get("max_seq").AsInt32(m->default_max_seq);
    RPG_CHECK(max_seq > 0 && max_seq <= m->max_pos,
              "session_json.max_seq=%d 必须在 (0, %d] 内", max_seq, m->max_pos);

    auto s = std::make_unique<Session>();
    s->model = model;
    s->kv = std::make_unique<rpg::KvCache>(m->model->config(), max_seq);

    s->max_tokens = cfg.Get("max_tokens").AsInt32(kDefaultMaxTokens);
    RPG_CHECK(s->max_tokens > 0, "session_json.max_tokens 必须为正（收到 %d）",
              s->max_tokens);
    // 缓存里至少得留一个位置给生成结果，否则这个会话永远产不出东西。
    // 这里收窄而不是报错：max_seq 偏小是配置问题，不该让会话建不出来。
    if (s->max_tokens >= max_seq) {
      RPG_LOG_WARN("session_json.max_tokens=%d 相对 max_seq=%d 过大，已收到 %d",
                   s->max_tokens, max_seq, max_seq - 1);
      s->max_tokens = max_seq - 1;
    }

    const rpg::Json stop = cfg.Get("stop");
    if (stop.IsArray()) {
      for (const rpg::Json& e : stop.elements()) {
        if (e.IsString()) s->stop.push_back(e.AsString());
      }
    }

    // 会话级默认采样参数。temperature <= 0 = 贪心，是默认值 —— 保证老请求
    // （不传 temperature）与 M4 的行为逐位一致。
    s->temperature = cfg.Get("temperature").AsFloat(s->temperature);
    if (s->temperature < 0.0f) s->temperature = 0.0f;
    if (s->temperature > kMaxTemperature) {
      RPG_LOG_WARN("session_json.temperature=%.3f 超过上限 %.1f，已钳制",
                   s->temperature, kMaxTemperature);
      s->temperature = kMaxTemperature;
    }
    s->seed = static_cast<uint64_t>(
        cfg.Get("seed").AsInt(static_cast<int64_t>(s->seed)));

    RPG_CHECK(g_next_session > 0, "会话句柄已耗尽（进程内已创建 %d 个）",
              g_next_session - 1);
    const int32_t h = g_next_session++;
    m->sessions.push_back(h);
    g_sessions.emplace(h, std::move(s));

    RPG_LOG_INFO("已创建会话 #%d（模型 #%d，缓存容量 %d，约 %.0f MiB）", h, model,
                 max_seq,
                 static_cast<double>(max_seq) *
                     m->model->config().n_layers * m->model->config().n_kv_heads *
                     m->model->config().head_dim * 8 / 1048576.0);
    return h;
  }
  RPG_ABI_CATCH(-1)
}

void rpg_engine_session_destroy(int32_t session) {
  try {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto it = g_sessions.find(session);
    if (it == g_sessions.end()) return;

    const auto mit = g_models.find(it->second->model);
    if (mit != g_models.end()) {
      auto& v = mit->second.sessions;
      v.erase(std::remove(v.begin(), v.end(), session), v.end());
    }
    g_sessions.erase(it);
  }
  RPG_ABI_CATCH_VOID()
}

int32_t rpg_engine_session_reset(int32_t session) {
  try {
    rpg::SetLastError("");
    Session* s = RequireSession(session);
    // Reset 只把长度归零，内存留着 —— 同一房间开新一局时不必重新分配那几百 MB。
    s->kv->Reset();
    s->cached_ids.clear();
    return RPG_ENGINE_OK;
  }
  RPG_ABI_CATCH(-1)
}

char* rpg_engine_generate(int32_t session, const char* request_json) {
  try {
    rpg::SetLastError("");
    Session* s = RequireSession(session);
    ModelEntry* m = RequireModel(s->model);
    const rpg::Model& model = *m->model;
    const rpg::ModelConfig& c = model.config();
    const rpg::Tokenizer& tok = model.tokenizer();

    // ---------------------------------------------------------- 1. 解析请求
    RPG_CHECK(request_json != nullptr && *request_json != '\0', "request_json 为空");
    std::string perr;
    const rpg::Json req = rpg::Json::Parse(request_json, &perr);
    RPG_CHECK(perr.empty(), "request_json 解析失败：%s", perr.c_str());
    RPG_CHECK(req.IsObject(), "request_json 必须是 JSON 对象");

    const std::vector<rpg::ChatMessage> msgs =
        rpg::ParseMessages(req.Get("messages"));
    RPG_CHECK(!msgs.empty(), "messages 为空：至少需要一条消息");
    // 工具列表为空数组时，渲染走的是"无工具"分支（与官方模板一致）。
    const rpg::Json tools = rpg::ParseTools(req.Get("tools"));

    // ---------------------------------------------------------- 2. 渲染 + 分词
    rpg::ChatTemplateOptions topt;
    topt.add_generation_prompt = true;
    // 缺省 = 思考模式（官方模板只在显式 false 时补空思考块）。
    topt.enable_thinking = req.Get("enable_thinking").AsBool(true);

    const std::string prompt = rpg::RenderChatml(msgs, tools, topt);
    std::vector<int32_t> ids = tok.Encode(prompt, /*parse_special=*/true);
    if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
    const int64_t total = static_cast<int64_t>(ids.size());
    RPG_CHECK(total > 0, "渲染后的 prompt 编码为空");

    // ---------------------------------------------------------- 3. 与缓存对前缀
    // 宿主每轮把**完整 messages** 传进来，所以本轮 prompt 通常是"上轮 prompt
    // + 新回复 + 新消息"。找出公共前缀，只对新增的部分做前向。
    int64_t lcp = 0;
    const int64_t cached = static_cast<int64_t>(s->cached_ids.size());
    RPG_CHECK(cached == s->kv->len(),
              "内部状态不一致：cached_ids=%lld 而 kv.len=%lld", cached,
              static_cast<int64_t>(s->kv->len()));
    while (lcp < cached && lcp < total && s->cached_ids[lcp] == ids[lcp]) ++lcp;

    // 至少要前向一个位置，才有"下一个 token 的分布"可用 ——
    // 全前缀命中（宿主原样重发同一份请求）时回退一个位置重算那一个 token。
    // 这里刻意不去缓存上次的 logits 行：多存一份状态就多一处要同步的地方，
    // 而代价只是重算一个位置（约 0.1 秒）。
    if (lcp == total) lcp = total - 1;
    const int64_t rows = total - lcp;  // 本次要前向的位置数（≥ 1）

    // ---------------------------------------------------------- 4. 生成长度与容量
    int32_t max_tokens = req.Get("max_tokens").AsInt32(s->max_tokens);
    if (max_tokens <= 0) max_tokens = s->max_tokens;

    // 采样参数：请求优先，其次会话默认（temperature <= 0 = 贪心）。温度钳到
    // [0, 上限] —— 旋钮拧到底也不会把模型推进"胡言乱语区"。
    float temperature = req.Get("temperature").AsFloat(s->temperature);
    if (temperature < 0.0f) temperature = 0.0f;
    if (temperature > kMaxTemperature) {
      RPG_LOG_WARN("temperature=%.3f 超过上限 %.1f，已钳制", temperature,
                   kMaxTemperature);
      temperature = kMaxTemperature;
    }
    const uint64_t seed = static_cast<uint64_t>(
        req.Get("seed").AsInt(static_cast<int64_t>(s->seed)));
    // **所有会失败的条件检查都在这里做完**，之后才动 cache —— 否则一次失败
    // 会把会话留在"cached_ids 与 kv.len 对不上"的状态里，下一次调用直接报
    // 内部状态不一致（这个顺序是 M4 第一次跑红时定下来的）。
    RPG_CHECK(total + max_tokens + kSeqMargin <= s->kv->max_seq(),
              "序列超长：prompt " RPG_SZ " + 生成 %d + 余量 %lld 超过会话容量 %lld"
              "（调大 session_json.max_seq，或先 session_reset）",
              RPG_SZ_ARG(total), max_tokens, static_cast<long long>(kSeqMargin),
              static_cast<long long>(s->kv->max_seq()));

    // ---------------------------------------------------------- 5. prefill
    if (lcp < s->kv->len()) {
      // 回退到 lcp：kv 里 [0, lcp) 的 K/V 与位置一一对应，内容仍然有效
      // （同一位置的 K/V 只由它自己的前缀决定），所以归零长度再设回 lcp 即可，
      // 不必重算前缀。
      s->kv->Reset();
    }
    s->kv->SetLen(lcp);

    std::vector<float> logits;
    try {
      rpg::ForwardFull(model, ids.data() + lcp, rows, &s->scratch, &logits,
                       rpg::ForwardOptions(), s->kv.get());
      RPG_CHECK(static_cast<int64_t>(logits.size()) == rows * c.vocab,
                "前向产出的 logits 形状不对");
      RPG_CHECK(s->kv->len() == total, "prefill 后 kv.len=%lld，应为 " RPG_SZ,
                static_cast<long long>(s->kv->len()), RPG_SZ_ARG(total));
      s->cached_ids.assign(ids.begin(), ids.end());
    } catch (...) {
      // 前向中途失败（例如 OOM）会把 cache 留在一个"说不清长度"的状态。
      // 宁可从零重来，也不能让下一次调用拿着错的 cached_ids 去复用前缀。
      s->kv->Reset();
      s->cached_ids.clear();
      throw;
    }

    // ---------------------------------------------------------- 6. 停止 token
    std::vector<int32_t> stop_ids;
    if (c.eos_id >= 0) stop_ids.push_back(c.eos_id);
    const int32_t im_end = tok.IdOf("<|im_end|>");
    if (im_end >= 0 && !ContainsId(stop_ids, im_end)) stop_ids.push_back(im_end);
    AppendStopStrings(tok, req.Get("stop"), &stop_ids);
    for (const std::string& sp : s->stop) AppendStopString(tok, sp, &stop_ids);

    // ---------------------------------------------------------- 7. 生成
    rpg::GenerateOptions gopt;
    gopt.max_tokens = max_tokens;
    gopt.stop_ids = stop_ids;
    gopt.sample.temperature = temperature;
    gopt.sample.top_k = kDefaultTopK;
    gopt.sample.top_p = kDefaultTopP;
    gopt.seed = seed;

    std::vector<int32_t> gen;
    const float* last = logits.data() + (rows - 1) * c.vocab;
    rpg::Generate(model, s->kv.get(), &s->scratch, last, gopt, &gen);
    s->cached_ids.insert(s->cached_ids.end(), gen.begin(), gen.end());

    // ---------------------------------------------------------- 8. 解码 + 解析
    // **不加 skip_special**：<tool_call> / </tool_call> 自己就是特殊 token，
    // 跳过它们会把结构标记一起删掉（DESIGN §16.4）。控制标记由解析器清。
    const std::string raw = tok.Decode(gen, /*skip_special=*/false);
    const rpg::ParsedOutput parsed = rpg::ParseModelOutput(raw);

    // ---------------------------------------------------------- 9. 拼响应
    std::string finish;
    if (!parsed.tool_calls.empty()) {
      finish = "tool_calls";
    } else if (parsed.unterminated_tool_call || parsed.unterminated_think) {
      // 被 max_tokens 截断在半截结构里：宿主据此决定重试还是当失败。
      finish = "length";
    } else if (static_cast<int32_t>(gen.size()) >= max_tokens &&
               (gen.empty() || !ContainsId(stop_ids, gen.back()))) {
      finish = "length";
    } else {
      finish = "stop";
    }

    rpg::Json resp = rpg::Json::Object();
    resp.Set("text", parsed.text);

    rpg::Json calls = rpg::Json::Array();
    for (const rpg::ParsedToolCall& pc : parsed.tool_calls) {
      rpg::Json one = rpg::Json::Object();
      one.Set("name", pc.name);
      // 参数不是合法对象时留 null —— 引擎只解析、不判断合法性，宿主拿到 null
      // 就当作"这次调用不可用"（parsed.ok 的语义）。
      one.Set("arguments", pc.ok ? pc.arguments : rpg::Json());
      calls.PushBack(std::move(one));
    }
    resp.Set("tool_calls", std::move(calls));
    resp.Set("finish_reason", finish);

    rpg::Json usage = rpg::Json::Object();
    usage.Set("prompt_tokens", total);
    usage.Set("completion_tokens", static_cast<int64_t>(gen.size()));
    resp.Set("usage", std::move(usage));

    return DupForAbi(JsonToAbiString(resp));
  }
  RPG_ABI_CATCH(nullptr)
}

void rpg_engine_free(char* buf) { delete[] buf; }

char* rpg_engine_model_info(int32_t model) {
  try {
    rpg::SetLastError("");
    ModelEntry* m = RequireModel(model);
    const rpg::ModelConfig& c = m->model->config();

    rpg::Json o = rpg::Json::Object();
    o.Set("arch", c.arch);
    o.Set("name", c.name);
    o.Set("n_layers", c.n_layers);
    o.Set("hidden", c.hidden);
    o.Set("n_heads", c.n_heads);
    o.Set("n_kv_heads", c.n_kv_heads);
    o.Set("head_dim", c.head_dim);
    o.Set("intermediate", c.intermediate);
    o.Set("vocab", c.vocab);
    o.Set("context_length", c.context_length);
    o.Set("rope_theta", static_cast<double>(c.rope_theta));
    o.Set("tied_embeddings", c.tied_embeddings);
    o.Set("path", m->model->path());
    o.Set("max_pos", m->max_pos);
    o.Set("default_max_seq", m->default_max_seq);
    o.Set("weight_bytes", m->model->weight_bytes());
    o.Set("sessions", static_cast<int64_t>(m->sessions.size()));

    return DupForAbi(JsonToAbiString(o));
  }
  RPG_ABI_CATCH(nullptr)
}

}  // extern "C"
