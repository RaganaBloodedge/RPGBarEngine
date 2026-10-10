// RPGBarEngine —— 最小 JSON
//
// 用在哪：引擎自己的输入输出全是 JSON ——
//   * 宿主传进来的 config_json / request_json（messages + tools + 采样参数）；
//   * 引擎返回的响应（text / tool_calls / usage）；
//   * 从 GGUF 里读出来的配置字符串。
//
// 为什么不引 nlohmann/json：这个库要编进游戏本体，"多一个头文件依赖"意味着
// 宿主的编译环境也要跟着迁就，而且它的异常类型会穿到我们的错误处理里。
// 我们只用到很小的一个子集（对象/数组/字符串/数字/布尔/null），自己写更可控，
// 也能给"缺字段就落默认值"这种行为量身定制接口。
//
// 不支持：注释、尾随逗号（严格按 RFC 8259）。

#ifndef RPGBAR_JSON_H_
#define RPGBAR_JSON_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rpg {

class Json {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : type_(Type::kBool), bool_(b) {}
  Json(int v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(int64_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(uint32_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(uint64_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(float v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(double v) : type_(Type::kNumber), num_(v) {}
  Json(const char* s) : type_(Type::kString), str_(s) {}
  Json(std::string s) : type_(Type::kString), str_(std::move(s)) {}

  static Json Array() {
    Json j;
    j.type_ = Type::kArray;
    return j;
  }
  static Json Object() {
    Json j;
    j.type_ = Type::kObject;
    return j;
  }

  Type type() const { return type_; }
  bool IsNull() const { return type_ == Type::kNull; }
  bool IsBool() const { return type_ == Type::kBool; }
  bool IsNumber() const { return type_ == Type::kNumber; }
  bool IsString() const { return type_ == Type::kString; }
  bool IsArray() const { return type_ == Type::kArray; }
  bool IsObject() const { return type_ == Type::kObject; }

  // ---- 取值（带默认值：JSON 里缺字段是常态，调用方不该为每个字段写分支）----
  bool AsBool(bool def = false) const {
    return type_ == Type::kBool ? bool_ : def;
  }
  double AsDouble(double def = 0.0) const {
    return type_ == Type::kNumber ? num_ : def;
  }
  float AsFloat(float def = 0.0f) const {
    return type_ == Type::kNumber ? static_cast<float>(num_) : def;
  }
  int64_t AsInt(int64_t def = 0) const {
    return type_ == Type::kNumber ? static_cast<int64_t>(num_) : def;
  }
  int AsInt32(int def = 0) const { return static_cast<int>(AsInt(def)); }
  const std::string& AsString() const { return str_; }
  std::string AsString(const std::string& def) const {
    return type_ == Type::kString ? str_ : def;
  }

  // ---- 对象 ----
  bool Has(const std::string& key) const {
    return type_ == Type::kObject && Find(key) >= 0;
  }
  const Json& operator[](const std::string& key) const;
  Json& operator[](const std::string& key);
  // 取不到就返回一个静态 null，方便链式取值。
  Json Get(const std::string& key) const {
    const int i = Find(key);
    return i >= 0 ? obj_[static_cast<size_t>(i)].second : Json();
  }
  void Set(const std::string& key, Json v);
  const std::vector<std::pair<std::string, Json>>& items() const {
    return obj_;
  }

  // ---- 数组 ----
  size_t size() const {
    return type_ == Type::kArray ? arr_.size() : obj_.size();
  }
  const Json& At(size_t i) const { return arr_[i]; }
  Json GetAt(size_t i) const {
    return (type_ == Type::kArray && i < arr_.size()) ? arr_[i] : Json();
  }
  void PushBack(Json v) {
    if (type_ != Type::kArray) {
      type_ = Type::kArray;
      arr_.clear();
    }
    arr_.push_back(std::move(v));
  }
  const std::vector<Json>& elements() const { return arr_; }

  // ---- 序列化 / 解析 ----
  std::string Dump(bool pretty = false, int indent = 0) const;

  // 解析失败返回 null 类型，并把原因写进 err（err 可为 nullptr）。
  static Json Parse(const std::string& text, std::string* err = nullptr);
  static bool ParseInto(const std::string& text, Json* out, std::string* err);

  // 便捷：把字符串包成 JSON 字符串字面量的转义形式（返回带引号的内容）。
  static std::string EscapeString(const std::string& s);

 private:
  int Find(const std::string& key) const;

  Type type_ = Type::kNull;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  std::vector<Json> arr_;
  std::vector<std::pair<std::string, Json>> obj_;
};

}  // namespace rpg

#endif  // RPGBAR_JSON_H_
